"""Unit tests for the CLAMP per-slice optimizer, with a stub for SAM."""

import numpy as np

from ..processing.clamp import (
    drop_stray_components,
    inner_loop,
    largest_cc_2d,
    membrane_trim,
    optimize_slice,
    segment_best,
)

SHAPE = (100, 100)


def rect(y1, y2, x1, x2):
    m = np.zeros(SHAPE, bool)
    m[y1:y2, x1:x2] = True
    return m


def growing_predict_fn(grow=2):
    """Stub SAM: returns the prompt box widened by ``grow`` px per side."""

    def predict_fn(box, mask_input):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m = rect(
            max(y1 - grow, 0),
            y2 + grow + 1,
            max(x1 - grow, 0),
            x2 + grow + 1,
        )
        return m[None], np.array([1.0])

    return predict_fn


def test_segment_best_picks_candidate_closest_to_reference():
    near, far = rect(10, 20, 10, 20), rect(60, 70, 60, 70)

    def predict_fn(box, mask_input):
        return np.stack([far, near]), np.array([0.9, 0.1])

    assert (segment_best(predict_fn, None, near, None) == near).all()
    # without a reference, SAM's own top-scoring candidate wins
    assert (segment_best(predict_fn, None, None, None) == far).all()


def test_inner_loop_stops_when_converged():
    def predict_fn(box, mask_input):
        return rect(10, 20, 10, 20)[None], np.array([1.0])

    masks = inner_loop(predict_fn, rect(10, 20, 10, 20))
    assert len(masks) == 2  # initial mask + one unchanged iteration


def test_inner_loop_respects_iteration_cap():
    masks = inner_loop(
        growing_predict_fn(1), rect(40, 50, 40, 50), max_iter=4
    )
    assert len(masks) == 5  # initial mask + max_iter iterations


def test_drop_stray_components_keeps_specks():
    body, stray, speck = (
        rect(0, 40, 0, 40),
        rect(60, 90, 60, 90),
        rect(50, 52, 0, 2),
    )
    out = drop_stray_components(body | stray | speck)
    assert (out == (body | speck)).all()


def test_largest_cc_2d():
    body, speck = rect(0, 40, 0, 40), rect(50, 52, 0, 2)
    assert (largest_cc_2d(body | speck) == body).all()
    assert not largest_cc_2d(np.zeros(SHAPE, bool)).any()


def test_membrane_trim_cuts_at_ridge():
    mask = rect(10, 50, 0, 80)
    core = rect(10, 50, 0, 30)
    ridge = np.zeros(SHAPE)
    ridge[:, 39:42] = 1.0
    out = membrane_trim(mask, core, ridge)
    assert out[10:50, :37].all()
    assert not out[:, 45:].any()
    # no membrane map: the gate is disabled
    assert (membrane_trim(mask, core, None) == mask).all()


def test_optimize_slice_inner_loop_grows_within_blowup():
    box, iter0 = [40, 40, 49, 49], rect(38, 52, 38, 52)
    fn = growing_predict_fn(2)
    outer_only = optimize_slice(fn, box, iter0, self_opt=False)
    assert (outer_only == iter0).all()
    refined = optimize_slice(fn, box, iter0, self_opt=True)
    assert (refined & iter0).sum() == iter0.sum()
    assert iter0.sum() < refined.sum() <= 3.0 * iter0.sum()


def test_optimize_slice_recovers_from_switch():
    prev, neighbour = rect(10, 30, 10, 30), rect(60, 80, 60, 80)

    def predict_fn(box, mask_input):
        m = neighbour if mask_input is None else prev
        return m[None], np.array([1.0])

    out = optimize_slice(predict_fn, [10, 10, 29, 29], prev, self_opt=False)
    assert (out == prev).all()


def test_optimize_slice_ends_object_when_it_vanishes():
    prev, neighbour = rect(10, 30, 10, 30), rect(60, 80, 60, 80)

    def predict_fn(box, mask_input):
        return neighbour[None], np.array([1.0])

    out = optimize_slice(predict_fn, [10, 10, 29, 29], prev, self_opt=False)
    assert not out.any()


def test_optimize_slice_shrink_is_not_a_switch():
    prev, shrunk = rect(10, 30, 10, 30), rect(15, 25, 15, 25)

    def predict_fn(box, mask_input):
        m = shrunk if mask_input is None else prev
        return m[None], np.array([1.0])

    out = optimize_slice(predict_fn, [15, 15, 24, 24], prev, self_opt=False)
    assert (out == shrunk).all()
