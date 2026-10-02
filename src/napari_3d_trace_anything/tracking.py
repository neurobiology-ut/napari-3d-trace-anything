import math
from importlib.resources import as_file, files

import cv2
import numpy as np

VIT_TRACKER_MODEL = "model/vit/object_tracking_vittrack_2023sep.onnx"
SUPPORTED_MOVEMENT_METHODS = ("ecc", "poc", "akaze", "tracker")


def _resolve_vit_model_path():
    resource = files("napari_3d_trace_anything").joinpath(VIT_TRACKER_MODEL)
    return as_file(resource)


def get_vit_tracker():
    """Create a fresh cv2.TrackerVit using the bundled ONNX model.

    cv2.TrackerVit_Params.net requires a real filesystem path
    (C++ fopen). Editable installs and unpacked wheels keep the file on
    disk; zip-imported environments are not supported.
    """
    with _resolve_vit_model_path() as model_path:
        params = cv2.TrackerVit_Params()
        params.net = str(model_path)
        return cv2.TrackerVit_create(params)


def _ensure_gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _to_uint8_gray(image):
    """Normalize an image to 8-bit single-channel for cv2 detectors.

    napari image layers are commonly float; AKAZE / medianBlur require
    CV_8U with arbitrary channel counts.
    """
    image = _ensure_gray(image)
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float64)
    max_val = float(img.max()) if img.size else 0.0
    if max_val <= 0 or not np.isfinite(max_val):
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip(img / max_val * 255.0, 0, 255).astype(np.uint8)


def compute_poc_displacement(img1, img2):
    """Phase-Only Correlation displacement between two images.

    Sign convention matches cv2.phaseCorrelate and compute_ecc_displacement:
    if img2 is img1 shifted by (+dx, +dy), this returns (+dx, +dy).

    For textureless / constant inputs (e.g. all-zero frames) the
    correlation response stays near zero everywhere and argmax would
    spuriously pick the (0, 0) corner; this is guarded by checking the
    peak height and returning (0, 0, 0) instead.

    Returns:
        (dx, dy, distance)
    """
    img1 = _to_uint8_gray(img1)
    img2 = _to_uint8_gray(img2)

    img1 = cv2.medianBlur(img1, 5)
    img2 = cv2.medianBlur(img2, 5)
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 1.0)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    h, w = img1.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    img1_w = img1 * hann
    img2_w = img2 * hann

    f1 = np.fft.fft2(img1_w)
    f2 = np.fft.fft2(img2_w)

    # F2 * conj(F1) yields a peak at +shift when img2 = shift(img1).
    cross_power = f2 * np.conj(f1)
    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)

    poc = np.fft.ifft2(cross_power_norm)
    poc = np.abs(np.fft.fftshift(poc))

    # Textureless inputs leave the correlation surface near zero; report
    # no movement rather than the spurious (0, 0) corner pick.
    if not np.isfinite(poc).all() or poc.max() < 1e-6:
        return 0, 0, 0.0

    peak_idx = np.unravel_index(np.argmax(poc), poc.shape)
    dy = peak_idx[0] - h // 2
    dx = peak_idx[1] - w // 2
    distance = float(np.sqrt(dx**2 + dy**2))
    return dx, dy, distance


def compute_ecc_displacement(img1, img2):
    """Enhanced Correlation Coefficient displacement.

    Returns:
        (dx, dy, distance, ecc_value). On failure: (0, 0, inf, 0.0).
    """
    img1 = _ensure_gray(img1).astype(np.float32)
    img2 = _ensure_gray(img2).astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )

    try:
        ecc_value, warp_matrix = cv2.findTransformECC(
            img1, img2, warp_matrix, warp_mode, criteria
        )
    except cv2.error:
        return 0, 0, float("inf"), 0.0

    dx = float(warp_matrix[0, 2])
    dy = float(warp_matrix[1, 2])
    distance = float(np.sqrt(dx**2 + dy**2))
    return dx, dy, distance, ecc_value


def compute_akaze_displacement(img1, img2):
    """AKAZE feature-matching displacement (median over top matches).

    AKAZE requires 8-bit input; float / uint16 images are normalized
    so float napari layers don't raise cv2.error.

    Returns:
        (dx, dy, distance). On insufficient matches: (0, 0, inf).
    """
    img1 = _to_uint8_gray(img1)
    img2 = _to_uint8_gray(img2)

    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0, 0, float("inf")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    if len(matches) < 3:
        return 0, 0, float("inf")

    matches = sorted(matches, key=lambda m: m.distance)
    good = matches[: min(50, len(matches))]

    disps = np.array(
        [
            (
                kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0],
                kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1],
            )
            for m in good
        ]
    )
    median_dx = float(np.median(disps[:, 0]))
    median_dy = float(np.median(disps[:, 1]))
    distance = float(np.sqrt(median_dx**2 + median_dy**2))
    return median_dx, median_dy, distance


def compute_tracker_displacement(
    ref_image, target_image, bbox_xywh, *, tracker=None
):
    """VitTracker-based displacement.

    Args:
        ref_image: RGB reference frame.
        target_image: RGB target frame.
        bbox_xywh: [x, y, width, height].
        tracker: Optional reusable ``cv2.TrackerVit`` instance. Pass one
            in to avoid re-reading the ONNX weights from disk on every
            call — re-``init()``ing the same tracker is supported.

    Returns:
        (dx, dy, distance, new_bbox_xywh). On failure: (0, 0, inf, None).
    """
    if tracker is None:
        tracker = get_vit_tracker()
    tracker.init(ref_image, tuple(bbox_xywh))
    ok, new_bbox = tracker.update(target_image)
    if not ok:
        return 0, 0, float("inf"), None

    old_cx = bbox_xywh[0] + bbox_xywh[2] / 2
    old_cy = bbox_xywh[1] + bbox_xywh[3] / 2
    new_cx = new_bbox[0] + new_bbox[2] / 2
    new_cy = new_bbox[1] + new_bbox[3] / 2
    dx = new_cx - old_cx
    dy = new_cy - old_cy
    distance = float(np.sqrt(dx**2 + dy**2))
    return dx, dy, distance, new_bbox


def extract_roi_around_box(image, bbox_xywh, scale=4.0):
    """Extract a scaled ROI centered on the bounding box.

    napari Shapes coordinates are float; bbox is int-cast here so that
    slicing succeeds for inputs like [40.0, 40.0, 20.0, 20.0].

    Returns:
        (roi_image, (x1, y1, x2, y2)) or (None, None) if invalid.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = (int(v) for v in bbox_xywh)
    cx = x + w // 2
    cy = y + h // 2
    roi_w = int(w * scale)
    roi_h = int(h * scale)

    x1 = max(0, cx - roi_w // 2)
    y1 = max(0, cy - roi_h // 2)
    x2 = min(img_w, cx + roi_w // 2)
    y2 = min(img_h, cy + roi_h // 2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def check_frame_movement(
    ref_image,
    target_image,
    bbox_xywh,
    threshold,
    scale=4.0,
    method="ecc",
    *,
    tracker=None,
):
    """Dispatcher: True if movement between frames exceeds threshold.

    Args:
        tracker: Optional reusable ``cv2.TrackerVit`` for ``method='tracker'``.
            Ignored for ECC / POC / AKAZE methods.

    Returns:
        (exceeds, distance, dx, dy, new_bbox_xywh)
        new_bbox_xywh is non-None only for method='tracker' on success.

    Raises:
        ValueError: if ``method`` is not one of SUPPORTED_MOVEMENT_METHODS.
    """
    if method not in SUPPORTED_MOVEMENT_METHODS:
        raise ValueError(
            f"unknown method {method!r}; "
            f"expected one of {SUPPORTED_MOVEMENT_METHODS}"
        )

    if method == "tracker":
        dx, dy, distance, new_bbox = compute_tracker_displacement(
            ref_image, target_image, bbox_xywh, tracker=tracker
        )
        exceeds = distance > threshold
        return exceeds, distance, dx, dy, (None if exceeds else new_bbox)

    roi_ref, _ = extract_roi_around_box(ref_image, bbox_xywh, scale)
    if roi_ref is None:
        return False, 0.0, 0, 0, None
    roi_tgt, _ = extract_roi_around_box(target_image, bbox_xywh, scale)
    if roi_tgt is None:
        return False, 0.0, 0, 0, None

    min_h = min(roi_ref.shape[0], roi_tgt.shape[0])
    min_w = min(roi_ref.shape[1], roi_tgt.shape[1])
    roi_ref = roi_ref[:min_h, :min_w]
    roi_tgt = roi_tgt[:min_h, :min_w]

    if method == "ecc":
        dx, dy, distance, _ = compute_ecc_displacement(roi_ref, roi_tgt)
    elif method == "akaze":
        dx, dy, distance = compute_akaze_displacement(roi_ref, roi_tgt)
    else:  # poc
        dx, dy, distance = compute_poc_displacement(roi_ref, roi_tgt)

    exceeds = distance > threshold
    return exceeds, distance, dx, dy, None


def compute_roi_for_segmentation(image_shape, boxes_xywh, max_size=1024):
    """Compute a crop region that contains every input box.

    The ROI targets ``max_size`` in each dimension to keep payloads small,
    but is expanded to fit all boxes when their union spans more than
    ``max_size``; in that case the ROI may grow up to the image size, and
    a crop strictly smaller than the full image is preferred over sending
    everything to the segmentation server.

    Returns:
        (needs_crop, x1, y1, x2, y2). When the image already fits in
        ``max_size`` or no boxes are supplied, returns
        (False, None, None, None, None).
    """
    img_h, img_w = image_shape[:2]
    if img_w <= max_size and img_h <= max_size:
        return False, None, None, None, None
    if not boxes_xywh:
        return False, None, None, None, None

    # Floor lower bounds and ceil upper bounds so that float coordinates
    # never shrink the union below the actual box extent.
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    for box in boxes_xywh:
        x, y, w, h = box
        all_x1.append(math.floor(x))
        all_y1.append(math.floor(y))
        all_x2.append(math.ceil(x + w))
        all_y2.append(math.ceil(y + h))

    box_x1 = min(all_x1)
    box_y1 = min(all_y1)
    box_x2 = max(all_x2)
    box_y2 = max(all_y2)

    center_x = (box_x1 + box_x2) // 2
    center_y = (box_y1 + box_y2) // 2

    roi_w = min(img_w, max_size)
    roi_h = min(img_h, max_size)

    roi_x1 = center_x - roi_w // 2
    roi_y1 = center_y - roi_h // 2
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    if roi_x1 < 0:
        roi_x2 -= roi_x1
        roi_x1 = 0
    if roi_y1 < 0:
        roi_y2 -= roi_y1
        roi_y1 = 0
    if roi_x2 > img_w:
        roi_x1 -= roi_x2 - img_w
        roi_x2 = img_w
    if roi_y2 > img_h:
        roi_y1 -= roi_y2 - img_h
        roi_y2 = img_h

    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)
    roi_x2 = min(img_w, roi_x2)
    roi_y2 = min(img_h, roi_y2)

    # Boxes whose union spans more than max_size cannot fit in a
    # max_size ROI; expand to contain them (still bounded by the image).
    if (
        box_x1 < roi_x1
        or box_x2 > roi_x2
        or box_y1 < roi_y1
        or box_y2 > roi_y2
    ):
        roi_x1 = max(0, min(roi_x1, box_x1))
        roi_y1 = max(0, min(roi_y1, box_y1))
        roi_x2 = min(img_w, max(roi_x2, box_x2))
        roi_y2 = min(img_h, max(roi_y2, box_y2))

    return True, int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)
