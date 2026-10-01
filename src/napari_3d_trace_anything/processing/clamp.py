"""CLAMP per-slice optimizer (Closed-Loop Auto-Mask Propagation).

CLAMP has two loops:

- **Outer loop** (propagation across slices) lives in the widget:
  ``TraceAnything._tracer`` steps through slices, and ``_predict`` turns
  the previous slice's mask into the prompt box for the current one
  (``create_boxes_list``) before calling :func:`optimize_slice`.
- **Inner loop** (closed-loop refinement within one slice) is
  :func:`inner_loop`: re-box from the current mask and re-segment until
  IoU between iterations exceeds 0.99 or ``MAX_ITER`` is reached. It runs
  only with ``self_opt=True``; ``self_opt=False`` is the outer-loop-only
  condition.

:func:`optimize_slice` also chooses among SAM's candidates by IoU with the
previous slice's mask and re-prompts when the mask jumps to a neighbouring
profile; the thresholds are the module constants below.

This module is pure: it takes a ``predict_fn`` callable and numpy arrays,
with no SAM/torch/IO dependency, so it is testable and reusable.
``predict_fn(box_xyxy, mask_input) -> (masks, scores)`` must return SAM's
multi-mask output: ``masks`` a stack of candidate masks and ``scores`` the
matching predicted-IoU scores. The caller is responsible for having set the
image on the underlying predictor before calling.
"""

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.filters import sato
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize

from .._utils import calculate_iou, create_box

# Parameters used in the paper. Override via keyword arguments.
COVER = 0.8  # min recall of the box-only mask for a self-opt iter to count
BLOWUP = 3.0  # max area ratio (self-opt iter / box-only) before rejecting
QMEM = 0.7  # in-mask ridge quantile above which pixels are cut (membrane)
R_MIN = 0.75  # recall-of-prev below which a switch is suspected
P_MIN = 0.6  # precision-of-prev below which a switch (vs shrink) is confirmed
VANISH_FLOOR = 0.4  # recall-of-prev below which the object is deemed gone
MAX_ITER = 10  # self-opt iteration cap
MIN_STRAY_AREA = 500  # drop non-largest components >= this (keep tiny specks)
MIN_TRIM_AREA = 200  # skip the membrane gate on tiny masks
TRIM_SPLIT_FRAC = 0.12  # min removed fraction for a split to count as a leak
TRIM_DILATE = 2  # dilation applied to the kept compartment before AND-ing


def membrane_map_from_image(roi):
    """Sato dark-ridge (membrane) map for a grayscale ROI in [0, 255]."""
    return sato(
        roi.astype(float) / 255.0, sigmas=[1, 2, 3], black_ridges=True
    )


def to_logits(mask):
    """Dense low-res prior for SAM's ``mask_input`` (256x256, +/-10 logits)."""
    if mask is None or not np.any(mask):
        return None
    lr = resize(mask.astype(float), (256, 256), order=0) * 20 - 10
    return lr[None].astype(np.float32)


def recall(mask, ref):
    """|mask & ref| / |ref| -- 1.0 when ref is empty."""
    ref = np.asarray(ref, bool)
    if not ref.any():
        return 1.0
    return float((np.asarray(mask, bool) & ref).sum()) / float(ref.sum())


def precision(mask, ref):
    """|mask & ref| / |mask| -- 0.0 when mask is empty."""
    mask = np.asarray(mask, bool)
    if not mask.sum():
        return 0.0
    return float((mask & np.asarray(ref, bool)).sum()) / float(mask.sum())


def segment_best(predict_fn, box, ref_mask, mask_input):
    """Base per-slice segmentation with reference-guided candidate choice.

    Runs SAM in multi-mask mode and returns the candidate with the highest
    IoU to ``ref_mask`` (the previous slice); when there is no reference,
    falls back to SAM's own top-scoring candidate.
    """
    masks, scores = predict_fn(box, mask_input)
    masks = np.asarray(masks) > 0
    if ref_mask is not None and np.any(ref_mask):
        ious = [calculate_iou(m, ref_mask) for m in masks]
        k = int(np.argmax(ious))
    else:
        k = int(np.argmax(scores))
    return masks[k]


def inner_loop(predict_fn, initial_mask, max_iter=MAX_ITER):
    """CLAMP inner loop: re-box and re-segment until convergence.

    Each iteration labels the current mask into connected components,
    re-segments each from its own bbox (reference = that component), and
    unions the results, until IoU converges or the cap is reached. Returns
    the list of masks (including ``initial_mask``) so the caller can pick.
    """
    masks = [initial_mask]
    cur = initial_mask
    prev = None
    it = 1
    while True:
        if prev is not None and calculate_iou(cur, prev) > 0.99:
            break
        prev = cur
        comps = sklabel(cur.astype(np.uint8))
        props = list(regionprops(comps))
        if not props:
            break
        new = np.zeros_like(cur)
        for p in props:
            new |= segment_best(
                predict_fn, create_box(p, 0.0), comps == p.label, None
            )
        cur = new
        masks.append(cur)
        it += 1
        if it > max_iter:
            break
    return masks


def membrane_trim(mask, core, membrane_map, qmem=QMEM):
    """Membrane gate: drop compartments separated from ``core`` by a ridge.

    Removes in-mask ridge pixels (> the ``qmem`` quantile of the membrane
    map inside the mask). If that splits the mask into multiple significant
    components, keep only the component(s) containing ``core``. A no-op when
    ``membrane_map`` is None (the caller disabled the gate).
    """
    if membrane_map is None:
        return mask
    a = int(mask.sum())
    if a < MIN_TRIM_AREA or not np.any(core):
        return mask
    thr = np.quantile(membrane_map[mask], qmem)
    cut = mask & (membrane_map <= thr)
    lab = sklabel(cut)
    if lab.max() <= 1:
        return mask
    core_ids = set(np.unique(lab[core & cut])) - {0}
    if not core_ids:
        return mask
    keep = np.isin(lab, list(core_ids))
    if (cut & ~keep & (lab > 0)).sum() < TRIM_SPLIT_FRAC * a:
        return mask  # nothing significant removed -> not a membrane crossing
    return binary_dilation(keep, iterations=TRIM_DILATE) & mask


def drop_stray_components(mask, min_stray=None):
    """Keep the largest component + tiny specks; drop *significant* strays.

    An axon is one profile per slice, so a large second blob is spurious.
    Tiny specks (< ``min_stray``) are kept because they enlarge the mask's
    bounding box, which seeds the next slice's prompt.
    """
    if min_stray is None:
        min_stray = MIN_STRAY_AREA
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    lab, n = sklabel(mask, return_num=True)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep_largest = int(sizes.argmax())
    # keep the largest, plus any component below the stray threshold
    keep = (lab == keep_largest) | np.isin(
        lab, np.flatnonzero((sizes > 0) & (sizes < min_stray))
    )
    return keep


def largest_cc_2d(mask):
    """Largest connected component of a 2D mask (output cleanup).

    Apply to the SAVED per-slice label, NOT to the mask that seeds the next
    slice's prompt box. :func:`drop_stray_components` deliberately keeps tiny
    box-inflating specks in the propagation mask (removing them starves
    growth -- see its docstring), but such a speck, or a mid-size stray below
    the drop threshold, still reads as a visible second blob in the saved
    label.
    Cleaning only the output removes these without touching the trajectory:
    the box was already derived from the speck-inflated mask, so a per-slice
    largest-component pass on the finished stack is exactly equivalent to
    having seeded from the full mask while saving only its largest component.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    lab, n = sklabel(mask, return_num=True)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def optimize_slice(
    predict_fn,
    box,
    ref_mask,
    membrane_map=None,
    self_opt=True,
    cover=COVER,
    blowup=BLOWUP,
    qmem=QMEM,
    r_min=R_MIN,
    p_min=P_MIN,
    vanish_floor=VANISH_FLOOR,
    max_iter=MAX_ITER,
    drop_strays=True,
):
    """Segment one slice with CLAMP (see the module docstring).

    Args:
        predict_fn: ``(box_xyxy, mask_input) -> (masks, scores)`` wrapping a
            SAM predictor whose image is already set for this slice/ROI.
        box: ``[x1, y1, x2, y2]`` prompt box in the ROI's coordinates.
        ref_mask: previous slice's mask in the ROI's coordinates, or None on
            the first slice.
        membrane_map: Sato ridge map for the ROI (see
            :func:`membrane_map_from_image`), or None to disable the gate.
        self_opt: when True run the inner loop with the coverage guard and
            membrane gate; when False only the outer loop is applied.
        drop_strays: when True drop significant non-largest
            connected components (spurious blobs) while keeping tiny specks.

    Returns:
        np.ndarray: boolean mask in the ROI's coordinates.
    """

    def finalize(mask):
        return drop_stray_components(mask) if drop_strays else mask

    iter0 = segment_best(predict_fn, box, ref_mask, None)  # box-only: free
    has_ref = ref_mask is not None and np.any(ref_mask)
    r0 = recall(iter0, ref_mask) if has_ref else 1.0
    p0 = precision(iter0, ref_mask) if has_ref else 1.0

    # a switch is a jump AWAY -> low recall AND low precision
    # of the previous mask. A legit shrink is low-recall but HIGH-precision
    # (the new mask is a subset of prev) and must NOT trigger recovery.
    if has_ref and r0 < r_min and p0 < p_min:
        rec = segment_best(predict_fn, box, ref_mask, to_logits(ref_mask))
        rr = recall(rec, ref_mask)
        # keep whichever candidate better matches the previous mask
        keep, keep_r = (rec, rr) if rr >= r0 else (iter0, r0)
        if keep_r < vanish_floor:
            return np.zeros_like(iter0)  # near-vanish -> object ended
        if self_opt:
            return finalize(
                membrane_trim(keep, keep & ref_mask, membrane_map, qmem)
            )
        return finalize(keep)

    if not self_opt:
        return finalize(iter0)

    a0 = int(iter0.sum())
    if a0 == 0:
        return iter0
    # Inner loop: keep the largest iteration that covers iter0 and is not a
    # gross blow-up, after the membrane gate.
    best = iter0
    candidates = sorted(
        inner_loop(predict_fn, iter0, max_iter), key=lambda x: -x.sum()
    )
    for m in candidates:
        if (m & iter0).sum() / a0 < cover or m.sum() > blowup * a0:
            continue
        trimmed = membrane_trim(m, iter0, membrane_map, qmem)
        if trimmed.sum() > best.sum():
            best = trimmed
    return finalize(best)
