import cv2
import numpy as np
import pytest

from napari_3d_trace_anything.tracking import (
    check_frame_movement,
    compute_akaze_displacement,
    compute_ecc_displacement,
    compute_poc_displacement,
    compute_roi_for_segmentation,
    extract_roi_around_box,
    get_vit_tracker,
)


def _checker_image(shift_x=0, shift_y=0):
    img = np.zeros((100, 100), dtype=np.uint8)
    y0, y1 = 30 + shift_y, 70 + shift_y
    x0, x1 = 30 + shift_x, 70 + shift_x
    img[y0:y1, x0:x1] = 255
    return img


def test_get_vit_tracker_loads():
    """Bundled ONNX resolves to a real path and creates a TrackerVit."""
    tracker = get_vit_tracker()
    assert isinstance(tracker, cv2.TrackerVit)


def test_compute_poc_displacement_no_movement():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    _dx, _dy, distance = compute_poc_displacement(img, img.copy())
    assert distance < 1.0


def test_compute_poc_displacement_with_shift():
    img1 = _checker_image()
    img2 = _checker_image(shift_x=5)
    dx, dy, distance = compute_poc_displacement(img1, img2)
    # img2 = img1 shifted by +5 in x; sign matches cv2.phaseCorrelate / ECC.
    assert dx == 5
    assert dy == 0
    assert distance >= 5.0


def test_compute_poc_displacement_sign_matches_ecc():
    img1 = _checker_image()
    img2 = _checker_image(shift_x=3, shift_y=2)
    poc_dx, poc_dy, _ = compute_poc_displacement(img1, img2)
    ecc_dx, ecc_dy, _, _ = compute_ecc_displacement(img1, img2)
    # Both methods should report displacement in the same direction.
    assert np.sign(poc_dx) == np.sign(ecc_dx)
    assert np.sign(poc_dy) == np.sign(ecc_dy)


def test_compute_ecc_displacement_identical_images():
    img = _checker_image()
    dx, dy, distance, _ = compute_ecc_displacement(img, img.copy())
    assert abs(dx) < 1.0
    assert abs(dy) < 1.0
    assert distance < 1.0


def test_compute_akaze_displacement_blank_returns_inf():
    blank = np.zeros((100, 100), dtype=np.uint8)
    _dx, _dy, distance = compute_akaze_displacement(blank, blank.copy())
    assert distance == float("inf")


def test_extract_roi_around_box_returns_scaled_region():
    image = np.zeros((100, 100), dtype=np.uint8)
    bbox = [40, 40, 20, 20]
    roi, coords = extract_roi_around_box(image, bbox, scale=2.0)
    assert roi is not None
    assert coords is not None
    assert roi.shape[0] >= 20
    assert roi.shape[1] >= 20


def test_extract_roi_around_box_clamps_at_corner():
    image = np.zeros((100, 100), dtype=np.uint8)
    bbox = [0, 0, 20, 20]
    roi, coords = extract_roi_around_box(image, bbox, scale=4.0)
    assert roi is not None
    # ROI is centered on box (cx=10, cy=10) with scale*20=80px width/height.
    # Clamped to image bounds: x1=0, y1=0, x2=50, y2=50.
    assert coords == (0, 0, 50, 50)
    assert roi.shape == (50, 50)


def test_extract_roi_around_box_accepts_float_bbox():
    """napari Shapes layer reports float coordinates; bbox must be int-cast."""
    image = np.zeros((100, 100), dtype=np.uint8)
    bbox = [40.0, 40.0, 20.0, 20.0]
    roi, coords = extract_roi_around_box(image, bbox, scale=2.0)
    assert roi is not None
    assert coords is not None
    assert all(isinstance(v, int) for v in coords)


def test_check_frame_movement_below_threshold():
    img = _checker_image()
    bbox = [40, 40, 20, 20]
    exceeds, _dist, _dx, _dy, new_bbox = check_frame_movement(
        img, img.copy(), bbox, threshold=10.0, method="poc"
    )
    assert not exceeds
    assert new_bbox is None


def test_check_frame_movement_above_threshold():
    img1 = _checker_image()
    img2 = _checker_image(shift_x=10, shift_y=10)
    bbox = [30, 30, 20, 20]
    exceeds, dist, _dx, _dy, _new = check_frame_movement(
        img1, img2, bbox, threshold=1.0, method="poc"
    )
    assert dist > 0
    assert exceeds


def test_compute_roi_for_segmentation_small_image_no_crop():
    needs_crop, *rest = compute_roi_for_segmentation(
        (500, 500), [[100, 100, 50, 50]], max_size=1024
    )
    assert needs_crop is False
    assert rest == [None, None, None, None]


def test_compute_roi_for_segmentation_large_image_crops():
    needs_crop, x1, y1, x2, y2 = compute_roi_for_segmentation(
        (2000, 2000), [[900, 900, 100, 100]], max_size=1024
    )
    assert needs_crop is True
    assert x2 - x1 == 1024
    assert y2 - y1 == 1024
    # ROI must contain the original box
    assert x1 <= 900 and x2 >= 1000
    assert y1 <= 900 and y2 >= 1000


def test_compute_roi_for_segmentation_asymmetric_image():
    needs_crop, x1, y1, x2, y2 = compute_roi_for_segmentation(
        (800, 2000), [[1500, 400, 50, 50]], max_size=1024
    )
    assert needs_crop is True
    # height is below max_size, so y range covers the full image
    assert (y1, y2) == (0, 800)
    assert x2 - x1 == 1024
    assert x1 <= 1500 and x2 >= 1550


def test_compute_roi_for_segmentation_boxes_span_exceeds_max_size():
    """When the box union spans more than max_size, the ROI expands to
    contain every box rather than falling back to the full image: a crop
    strictly smaller than the source is still a win."""
    img_h, img_w = 5000, 5000
    needs_crop, x1, y1, x2, y2 = compute_roi_for_segmentation(
        (img_h, img_w),
        [[100, 100, 50, 50], [4000, 4000, 50, 50]],
        max_size=1024,
    )
    assert needs_crop is True
    # ROI contains every input box.
    assert x1 <= 100 and x2 >= 4050
    assert y1 <= 100 and y2 >= 4050
    # And is strictly smaller than the full image.
    assert (x2 - x1, y2 - y1) != (img_w, img_h)


def test_compute_roi_for_segmentation_roi_stays_within_max_size():
    """When the box union fits inside max_size, the ROI is capped at it."""
    needs_crop, x1, y1, x2, y2 = compute_roi_for_segmentation(
        (5000, 5000),
        [[100, 100, 50, 50], [900, 900, 50, 50]],
        max_size=1024,
    )
    assert needs_crop is True
    assert x2 - x1 <= 1024
    assert y2 - y1 <= 1024


def test_compute_roi_for_segmentation_empty_boxes_returns_false():
    """min()/max() on empty input would raise; guard returns False instead."""
    needs_crop, *rest = compute_roi_for_segmentation(
        (5000, 5000), [], max_size=1024
    )
    assert needs_crop is False
    assert rest == [None, None, None, None]


def test_compute_roi_for_segmentation_float_box_not_truncated():
    """Float bbox upper bounds must be rounded up so the ROI covers them."""
    needs_crop, x1, y1, x2, y2 = compute_roi_for_segmentation(
        (5000, 5000),
        [[100.7, 100.2, 50.4, 50.9]],
        max_size=1024,
    )
    assert needs_crop is True
    # Upper bounds: ceil(100.7 + 50.4) = 152, ceil(100.2 + 50.9) = 152
    assert x2 >= 152
    assert y2 >= 152


def test_compute_akaze_displacement_accepts_float_image():
    """Float images must be normalized to uint8 inside AKAZE instead of
    raising cv2.error."""
    rng = np.random.default_rng(42)
    img1 = rng.random((100, 100)).astype(np.float64)
    img2 = rng.random((100, 100)).astype(np.float64)
    # Should not raise; precise dx/dy is irrelevant.
    _dx, _dy, distance = compute_akaze_displacement(img1, img2)
    assert distance >= 0 or distance == float("inf")


def test_compute_poc_displacement_textureless_input():
    """All-zero frames must not produce a spurious (-w/2, -h/2) reading."""
    blank = np.zeros((100, 100), dtype=np.uint8)
    dx, dy, distance = compute_poc_displacement(blank, blank.copy())
    assert (dx, dy, distance) == (0, 0, 0.0)


def test_check_frame_movement_rejects_unknown_method():
    img = _checker_image()
    with pytest.raises(ValueError, match="unknown method"):
        check_frame_movement(
            img, img.copy(), [40, 40, 20, 20], threshold=10.0, method="pco"
        )
