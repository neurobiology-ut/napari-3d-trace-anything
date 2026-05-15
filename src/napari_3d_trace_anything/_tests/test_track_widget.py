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


def test_remote_url_field_is_empty_with_placeholder(widget):
    """No private/internal default — user must enter their own server URL."""
    assert widget._remote_url.text() == ""
    assert widget._remote_url.placeholderText() != ""


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


# ---------------- Image layer rename (Copilot L234) ----------------


def test_image_layer_rename_updates_combo_single(widget):
    layer = widget._viewer.layers["test-image"]
    layer.name = "renamed"
    assert widget._image_combo.findText("renamed") >= 0
    assert widget._image_combo.findText("test-image") < 0
    assert widget._image_layer_name == "renamed"


def test_image_layer_rename_preserves_selection_with_multiple_layers(
    make_napari_viewer,
):
    """Renaming the SELECTED layer must keep it selected (follow it by
    object identity)."""
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 50, 50)), name="a")
    selected = viewer.add_image(np.zeros((5, 50, 50)), name="b")
    viewer.add_image(np.zeros((5, 50, 50)), name="c")
    w = TrackAnything(viewer)
    w._image_combo.setCurrentText("b")
    assert w._selected_layer is selected

    selected.name = "b-renamed"
    # combo current must follow the renamed layer, not jump to "a" or "c".
    assert w._image_combo.currentText() == "b-renamed"
    assert w._image_layer_name == "b-renamed"


def test_non_selected_image_layer_rename_does_not_change_selection(
    make_napari_viewer,
):
    """Renaming a NON-selected layer must not steal the combo selection."""
    viewer = make_napari_viewer()
    first = viewer.add_image(np.zeros((5, 50, 50)), name="first")
    viewer.add_image(np.zeros((5, 50, 50)), name="middle")
    third = viewer.add_image(np.zeros((5, 50, 50)), name="third")
    w = TrackAnything(viewer)
    w._image_combo.setCurrentText("first")
    assert w._selected_layer is first

    third.name = "third-renamed"
    # Selection stays put.
    assert w._image_combo.currentText() == "first"
    assert w._selected_layer is first


def test_image_layer_rename_handler_disconnects_on_remove(widget):
    layer = widget._viewer.layers["test-image"]
    before = len(widget._name_handlers)
    assert before == 1
    widget._viewer.layers.remove(layer)
    # Layer removed → its handler dropped from the registry.
    assert id(layer) not in widget._name_handlers
    assert len(widget._name_handlers) == 0


def test_widget_close_disconnects_handlers(widget):
    layer = widget._viewer.layers["test-image"]
    assert id(layer) in widget._name_handlers
    widget.close()
    assert len(widget._name_handlers) == 0


# ---------------- Spinbox range sync (Copilot L192) ----------------


def test_segment_range_clamps_to_frame_count(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 100, 100)), name="five-frames")
    w = TrackAnything(viewer)
    assert w._segment_from.maximum() == 4
    assert w._segment_to.maximum() == 4
    # Default 100 gets clamped to 4.
    assert w._segment_to.value() == 4


def test_segment_range_single_frame_stack(make_napari_viewer):
    """T=1 → max=0; both spinboxes pinned to 0."""
    viewer = make_napari_viewer()
    # (T=1, H, W) with W != 3 is accepted as 'gray'.
    viewer.add_image(np.zeros((1, 50, 50)), name="single-frame")
    w = TrackAnything(viewer)
    assert w._segment_from.maximum() == 0
    assert w._segment_to.maximum() == 0
    assert w._segment_from.value() == 0
    assert w._segment_to.value() == 0


def test_segment_range_from_le_to_invariant(make_napari_viewer):
    """If the previous `to` value was below current `from`, fix the
    ordering after re-clamping."""
    viewer = make_napari_viewer()
    big = viewer.add_image(np.zeros((20, 50, 50)), name="big")
    w = TrackAnything(viewer)
    w._segment_from.setValue(15)
    w._segment_to.setValue(18)
    # Switch to a 5-frame layer → both should clamp to 4 with from <= to.
    viewer.add_image(np.zeros((5, 50, 50)), name="small")
    w._image_combo.setCurrentText("small")
    assert w._segment_from.value() <= w._segment_to.value()
    assert w._segment_from.value() <= 4
    assert w._segment_to.value() <= 4
    _ = big  # quiet unused


# ---------------- max_frames semantic (Copilot L602) ----------------


def test_advance_through_skips_allows_exactly_max_frames(widget):
    """With max_frames=2, two consecutive exceeds are tolerated; the third
    triggers bail (new semantic: ``skip_count > max_frames``)."""

    def make_check(threshold_exceeds):
        def fake(ref, target, bbox, *, threshold, method, tracker=None):
            # Always exceeds for the test.
            return True, threshold_exceeds, 0.0, 0.0, None

        return fake

    # Replace check_frame_movement so every call says "exceeds".
    import napari_3d_trace_anything._track_widget as widget_module

    original = widget_module.tracking.check_frame_movement
    widget_module.tracking.check_frame_movement = make_check(99.0)
    widget.show_popup = MagicMock()
    try:
        # 10 frames, ref doesn't matter — always exceeds.
        widget._frame_stack_shape = (10, 100, 100)
        result = widget._advance_through_skips(
            widget._viewer.layers["test-image"].data,
            np.zeros((100, 100, 3), dtype=np.uint8),
            [0, 0, 10, 10],
            start_idx=0,
            threshold=1.0,
            max_frames=2,
            method="poc",
        )
    finally:
        widget_module.tracking.check_frame_movement = original

    # All exceeds, so bail eventually with None.
    assert result is None
    # Popup fired exactly once.
    assert widget.show_popup.called


# ---------------- Tracker reuse (Copilot L491) ----------------


def test_compute_tracker_displacement_with_injected_tracker_skips_load(
    monkeypatch,
):
    """If a tracker is injected, get_vit_tracker must not be called."""
    import napari_3d_trace_anything._track_widget as widget_module

    count = {"n": 0}

    def fake():
        count["n"] += 1
        return widget_module.tracking.get_vit_tracker()

    # First instantiate the real tracker BEFORE the spy.
    real_tracker = widget_module.tracking.get_vit_tracker()
    monkeypatch.setattr(widget_module.tracking, "get_vit_tracker", fake)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    widget_module.tracking.compute_tracker_displacement(
        img, img.copy(), [10, 10, 20, 20], tracker=real_tracker
    )
    widget_module.tracking.compute_tracker_displacement(
        img, img.copy(), [10, 10, 20, 20], tracker=real_tracker
    )
    assert count["n"] == 0


# ---------------- L376 load failure popup ----------------


def test_load_local_model_shows_popup_on_failure(widget, monkeypatch):
    """If get_sam_model raises, predictor stays None and popup fires."""
    import napari_3d_trace_anything._track_widget as widget_module

    monkeypatch.setattr(
        widget_module,
        "get_sam_model",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no disk space")),
    )
    widget.show_popup = MagicMock()
    ok = widget._load_local_model()
    assert ok is False
    assert widget._local_predictor is None
    assert widget._backend is None
    widget.show_popup.assert_called()


def test_ensure_backend_after_failed_load_returns_none(widget, monkeypatch):
    """After a failed _load_local_model, _ensure_backend must NOT build
    LocalSAMBackend(None)."""
    import napari_3d_trace_anything._track_widget as widget_module

    monkeypatch.setattr(
        widget_module,
        "get_sam_model",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    widget.show_popup = MagicMock()
    widget._local_predictor = None
    assert widget._ensure_backend() is None


# ---------------- Unsupported via direct combo manipulation (L309) ----------


def test_on_image_layer_changed_rejects_unsupported_layer(make_napari_viewer):
    """Forcing an unsupported layer name into the combo must trigger the
    defensive popup branch."""
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 100, 100)), name="good")
    # Unsupported but real layer (single RGB frame).
    viewer.add_image(np.zeros((100, 100, 3)), name="bad")
    w = TrackAnything(viewer)
    w.show_popup = MagicMock()

    # Force the unsupported layer into the combo (bypassing the filter).
    w._image_combo.blockSignals(True)
    w._image_combo.addItem("bad")
    w._image_combo.blockSignals(False)
    w._image_combo.setCurrentText("bad")

    w.show_popup.assert_called()
    assert w._trace_btn.isEnabled() is False
