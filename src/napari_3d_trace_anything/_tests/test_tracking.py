import cv2
import numpy as np

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
    dx, _dy, distance = compute_poc_displacement(img1, img2)
    assert abs(dx) > 0 or distance > 1.0


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
