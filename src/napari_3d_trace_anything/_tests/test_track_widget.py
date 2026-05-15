from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_3d_trace_anything import TrackAnything, sam_backends


@pytest.fixture
def widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((10, 100, 100)), name="test-image")
    return TrackAnything(viewer)


# ---------------- UI smoke ----------------


def test_widget_creates_layers_with_image_prefix(widget):
    names = [layer.name for layer in widget._viewer.layers]
    assert "Track-Box-test-image" in names
    assert "Track-Label-test-image" in names


def test_widget_box_layer_does_not_clash_with_trace_anything(widget):
    """TraceAnything uses 'SAM-Box'; TrackAnything must use 'Track-Box-...'."""
    names = [layer.name for layer in widget._viewer.layers]
    assert "SAM-Box" not in names
    assert any(n.startswith("Track-Box-") for n in names)


def test_widget_default_backend_is_local(widget):
    assert widget._local_radio.isChecked()
    assert not widget._remote_radio.isChecked()


def test_backend_toggle_enables_correct_subsection(widget):
    assert widget._local_box.isEnabled()
    assert not widget._remote_box.isEnabled()

    widget._remote_radio.setChecked(True)
    assert not widget._local_box.isEnabled()
    assert widget._remote_box.isEnabled()


def test_skip_group_collapsible_default_off(widget):
    assert widget._skip_group.isCheckable()
    assert widget._skip_group.isChecked() is False


def test_segment_group_collapsible_default_off(widget):
    assert widget._segment_group.isCheckable()
    assert widget._segment_group.isChecked() is False


def test_debug_mode_unchecked_by_default(widget):
    assert widget._debug_mode.isChecked() is False


def test_trace_button_disabled_when_no_image(make_napari_viewer):
    """Combo populated only after Image layers are present."""
    viewer = make_napari_viewer()
    w = TrackAnything(viewer)
    assert w._trace_btn.isEnabled() is False
    assert w._image_combo.count() == 0


# ---------------- Image layer combo ----------------


def test_image_combo_picks_up_added_layer(make_napari_viewer):
    viewer = make_napari_viewer()
    w = TrackAnything(viewer)
    viewer.add_image(np.random.random((5, 50, 50)), name="late-add")
    assert w._image_combo.findText("late-add") >= 0
    assert w._trace_btn.isEnabled()


def test_image_combo_drops_removed_layer(widget):
    viewer = widget._viewer
    viewer.layers.remove("test-image")
    assert widget._image_combo.findText("test-image") < 0


def test_image_combo_ignores_non_image_layers(widget):
    """Labels/Shapes layers must not appear in the Image combo."""
    widget._viewer.add_labels(
        np.zeros((10, 100, 100), dtype="uint8"), name="not-an-image"
    )
    items = [
        widget._image_combo.itemText(i)
        for i in range(widget._image_combo.count())
    ]
    assert "not-an-image" not in items


# ---------------- Box layer / shapes ----------------


def test_reset_box_clears_data(widget):
    widget._box_layer.add_rectangles(
        [np.array([[0, 10, 10], [0, 10, 50], [0, 50, 50], [0, 50, 10]])]
    )
    assert len(widget._box_layer.data) > 0
    widget._reset_box()
    assert len(widget._box_layer.data) == 0


def test_get_bounding_box_single(widget):
    widget._box_layer.add_rectangles(
        [np.array([[0, 10, 10], [0, 10, 50], [0, 50, 50], [0, 50, 10]])]
    )
    coords = widget._get_bounding_box(0)
    assert coords == [[10, 10, 40, 40]]


def test_get_bounding_box_returns_none_when_multiple(widget):
    widget._box_layer.add_rectangles(
        [
            np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]]),
            np.array([[0, 40, 40], [0, 40, 50], [0, 50, 50], [0, 50, 40]]),
        ]
    )
    assert widget._get_bounding_box(0, silent=True) is None


def test_get_bounding_box_returns_none_when_missing(widget):
    assert widget._get_bounding_box(5, silent=True) is None


# ---------------- Backend wiring ----------------


def test_ensure_backend_remote_builds_remote_backend(widget):
    widget._remote_radio.setChecked(True)
    widget._remote_url.setText("http://stub")
    backend = widget._ensure_backend()
    assert isinstance(backend, sam_backends.RemoteSAMBackend)
    assert backend.url == "http://stub"


def test_ensure_backend_remote_empty_url_returns_none(widget):
    widget._remote_radio.setChecked(True)
    widget._remote_url.setText("   ")
    widget.show_popup = MagicMock()
    assert widget._ensure_backend() is None
    widget.show_popup.assert_called()


def test_ensure_backend_remote_rebuilds_on_url_change(widget):
    widget._remote_radio.setChecked(True)
    widget._remote_url.setText("http://a")
    a = widget._ensure_backend()
    widget._remote_url.setText("http://b")
    b = widget._ensure_backend()
    assert a is not b
    assert b.url == "http://b"


def test_ensure_backend_local_uses_loaded_predictor(widget):
    """A pre-loaded predictor must be wrapped without re-loading SAM."""
    fake_predictor = MagicMock()
    fake_predictor.set_image = MagicMock()
    fake_predictor.predict = MagicMock(
        return_value=(
            np.ones((1, 100, 100), dtype=bool),
            np.array([1.0]),
            None,
        )
    )
    widget._local_predictor = fake_predictor
    widget._local_radio.setChecked(True)
    backend = widget._ensure_backend()
    assert isinstance(backend, sam_backends.LocalSAMBackend)


def test_backend_toggle_clears_cached_backend(widget):
    widget._remote_radio.setChecked(True)
    widget._remote_url.setText("http://stub")
    widget._ensure_backend()
    assert widget._backend is not None
    widget._local_radio.setChecked(True)
    # After toggle, cached backend is dropped so it can be rebuilt.
    assert widget._backend is None


# ---------------- _segment_frame ----------------


def test_segment_frame_paints_mask_into_full_image(widget):
    """No-crop path: mask returned for the requested frame index."""
    fake = MagicMock()
    fake.prepare = MagicMock()
    fake.segment = MagicMock(return_value=np.ones((100, 100), dtype=bool))
    widget._backend = fake
    widget._ensure_backend = lambda: fake

    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    result = widget._segment_frame(image_rgb, [[10, 10, 20, 20]], 3)
    assert result is not None
    mask, idx = result
    assert idx == 3
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)
    assert mask.any()


def test_segment_frame_crops_large_image_and_restores(widget):
    """For 2000x800 input the segment is run on a crop, then pasted back."""
    fake = MagicMock()
    fake.prepare = MagicMock()
    # Return a mask matching whatever shape prepare was called with.
    captured = {}

    def fake_prepare(image, image_id):
        captured["shape"] = image.shape

    def fake_segment(box_xyxy):
        h, w = captured["shape"][:2]
        return np.ones((h, w), dtype=bool)

    fake.prepare.side_effect = fake_prepare
    fake.segment.side_effect = fake_segment
    widget._backend = fake
    widget._ensure_backend = lambda: fake

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    result = widget._segment_frame(image_rgb, [[1500, 400, 50, 50]], 7)
    assert result is not None
    mask, _ = result
    assert mask.shape == (800, 2000)
    # The mask must be confined to the cropped region (width 1024), not
    # the full image — verifies that the crop+restore math agrees.
    assert mask.sum() <= 1024 * 800


def test_segment_frame_returns_none_on_backend_exception(widget):
    fake = MagicMock()
    fake.prepare = MagicMock(side_effect=RuntimeError("nope"))
    widget._backend = fake
    widget._ensure_backend = lambda: fake
    widget.show_popup = MagicMock()

    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert widget._segment_frame(image_rgb, [[10, 10, 20, 20]], 0) is None
    widget.show_popup.assert_called()


def test_segment_frame_cache_key_includes_roi(widget):
    """Codex P1: same frame + different ROI must NOT share a cache slot.

    If image_id ignored the ROI, the second call would see image_id ==
    first call's image_id and the backend would short-circuit prepare(),
    leaving the stale crop in place while segment() ran against the new
    crop-relative box coordinates.
    """
    fake = MagicMock()
    captured = {}

    def fake_prepare(image, image_id):
        captured["shape"] = image.shape

    def fake_segment(box_xyxy):
        h, w = captured["shape"][:2]
        return np.ones((h, w), dtype=bool)

    fake.prepare.side_effect = fake_prepare
    fake.segment.side_effect = fake_segment
    widget._backend = fake
    widget._ensure_backend = lambda: fake

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    # Same frame index, two boxes far enough apart to produce different
    # 1024-wide crops.
    widget._segment_frame(image_rgb, [[100, 400, 50, 50]], 5)
    widget._segment_frame(image_rgb, [[1900, 400, 50, 50]], 5)

    assert fake.prepare.call_count == 2
    id1 = fake.prepare.call_args_list[0].kwargs["image_id"]
    id2 = fake.prepare.call_args_list[1].kwargs["image_id"]
    assert id1 != id2
    # The frame index is still recoverable from the key for callers that
    # want it.
    assert 5 in id1 and 5 in id2


def test_segment_frame_cache_key_stable_for_identical_roi(widget):
    """Identical ROI must produce identical image_id (so the local
    backend can still skip set_image)."""
    fake = MagicMock()
    fake.prepare = MagicMock()
    fake.segment = MagicMock(return_value=np.ones((800, 1024), dtype=bool))
    widget._backend = fake
    widget._ensure_backend = lambda: fake

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    widget._segment_frame(image_rgb, [[1500, 400, 50, 50]], 5)
    widget._segment_frame(image_rgb, [[1500, 400, 50, 50]], 5)

    id1 = fake.prepare.call_args_list[0].kwargs["image_id"]
    id2 = fake.prepare.call_args_list[1].kwargs["image_id"]
    assert id1 == id2


# ---------------- Image-layer shape validation (Codex P2) ----------------


def test_image_combo_filters_2d_image(make_napari_viewer):
    """A single 2D frame has no time axis to track; combo must omit it."""
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((100, 100), dtype=np.uint8), name="single-2d")
    w = TrackAnything(viewer)
    assert w._image_combo.findText("single-2d") < 0
    assert w._trace_btn.isEnabled() is False


def test_image_combo_filters_single_rgb_image(make_napari_viewer):
    """A single (H, W, 3) frame is rejected for the same reason."""
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((100, 100, 3), dtype=np.uint8), name="single-rgb"
    )
    w = TrackAnything(viewer)
    assert w._image_combo.findText("single-rgb") < 0


def test_widget_accepts_rgb_time_series(make_napari_viewer):
    """(T, H, W, 3) is supported; label layer must be (T, H, W)."""
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 100, 100, 3), dtype=np.uint8), name="rgb-ts")
    w = TrackAnything(viewer)
    assert w._image_combo.findText("rgb-ts") >= 0
    assert w._image_kind == "rgb"
    label = viewer.layers["Track-Label-rgb-ts"]
    # Label is 2D per frame, NOT (T, H, W, 3).
    assert label.data.shape == (5, 100, 100)


def test_widget_accepts_gray_with_channel(make_napari_viewer):
    """(T, H, W, 1) is supported; the channel suffix is stripped."""
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((5, 100, 100, 1), dtype=np.uint8), name="gray-ch"
    )
    w = TrackAnything(viewer)
    assert w._image_kind == "gray_ch"
    label = viewer.layers["Track-Label-gray-ch"]
    assert label.data.shape == (5, 100, 100)


def test_extract_frame_squeezes_gray_channel(make_napari_viewer):
    viewer = make_napari_viewer()
    images = np.arange(5 * 100 * 100, dtype=np.uint8).reshape(5, 100, 100, 1)
    viewer.add_image(images, name="gray-ch")
    w = TrackAnything(viewer)
    frame = w._extract_frame(images, 2)
    assert frame.shape == (100, 100)


def test_extract_frame_passes_rgb_through(make_napari_viewer):
    viewer = make_napari_viewer()
    images = np.zeros((5, 100, 100, 3), dtype=np.uint8)
    viewer.add_image(images, name="rgb-ts")
    w = TrackAnything(viewer)
    frame = w._extract_frame(images, 0)
    assert frame.shape == (100, 100, 3)


def test_extract_frame_gray_passthrough(widget):
    """(T, H, W) grayscale: extract_frame is a no-op on indexing."""
    images = widget._viewer.layers["test-image"].data
    frame = widget._extract_frame(images, 0)
    assert frame.shape == (100, 100)
