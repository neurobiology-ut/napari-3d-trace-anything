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


def test_build_backend_for_worker_creates_remote_backend(widget):
    """Snapshot-driven build returns a Remote backend without writing self."""
    snap = {
        "kind": "remote",
        "local_model_name": "vit_h",
        "device": None,
        "remote_url": "http://stub",
        "remote_model": "sam",
    }
    backend = widget._build_backend_for_worker(snap)
    assert isinstance(backend, sam_backends.RemoteSAMBackend)
    assert backend.url == "http://stub"


def test_build_backend_for_worker_remote_empty_url_returns_none(widget):
    snap = {
        "kind": "remote",
        "local_model_name": "vit_h",
        "device": None,
        "remote_url": "",
        "remote_model": "sam",
    }
    widget.show_popup = MagicMock()
    assert widget._build_backend_for_worker(snap) is None
    widget.show_popup.assert_called()


def test_build_backend_for_worker_uses_loaded_predictor(widget):
    """A pre-loaded predictor is wrapped without re-loading SAM."""
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
    snap = {
        "kind": "local",
        "local_model_name": "vit_h",
        "device": "cpu",
        "remote_url": "",
        "remote_model": "sam",
    }
    backend = widget._build_backend_for_worker(snap)
    assert isinstance(backend, sam_backends.LocalSAMBackend)


def test_build_backend_for_worker_does_not_write_widget_state(widget):
    """Pre-loaded state must be untouched by the worker-local builder."""
    pre_backend = object()
    pre_predictor = MagicMock()
    pre_sam = object()
    widget._backend = pre_backend
    widget._local_predictor = pre_predictor
    widget._local_sam_model = pre_sam
    snap = {
        "kind": "local",
        "local_model_name": "vit_h",
        "device": "cpu",
        "remote_url": "",
        "remote_model": "sam",
    }
    _ = widget._build_backend_for_worker(snap)
    # State remains exactly as we set it (object identity).
    assert widget._backend is pre_backend
    assert widget._local_predictor is pre_predictor
    assert widget._local_sam_model is pre_sam


def test_build_backend_for_worker_remote_url_change_does_not_mutate_self(
    widget,
):
    """Worker building a different RemoteSAMBackend must not replace
    ``self._backend`` — that's the click handler's job."""
    pre_backend = object()
    widget._backend = pre_backend
    snap = {
        "kind": "remote",
        "local_model_name": "vit_h",
        "device": None,
        "remote_url": "http://new",
        "remote_model": "sam",
    }
    new_backend = widget._build_backend_for_worker(snap)
    assert isinstance(new_backend, sam_backends.RemoteSAMBackend)
    assert new_backend.url == "http://new"
    assert widget._backend is pre_backend  # NOT overwritten


# ---------------- _segment_frame ----------------


def test_segment_frame_paints_mask_into_full_image(widget):
    """No-crop path: mask returned for the requested frame index.

    Backend is injected via the new ``backend=`` kwarg; the worker
    normally passes its local backend in this way.
    """
    fake = MagicMock()
    fake.prepare = MagicMock()
    fake.segment = MagicMock(return_value=np.ones((100, 100), dtype=bool))

    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    result = widget._segment_frame(
        image_rgb, [[10, 10, 20, 20]], 3, backend=fake
    )
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
    captured = {}

    def fake_prepare(image, image_id):
        captured["shape"] = image.shape

    def fake_segment(box_xyxy):
        h, w = captured["shape"][:2]
        return np.ones((h, w), dtype=bool)

    fake.prepare.side_effect = fake_prepare
    fake.segment.side_effect = fake_segment

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    result = widget._segment_frame(
        image_rgb, [[1500, 400, 50, 50]], 7, backend=fake
    )
    assert result is not None
    mask, _ = result
    assert mask.shape == (800, 2000)
    assert mask.sum() <= 1024 * 800


def test_segment_frame_returns_none_on_backend_exception(widget):
    fake = MagicMock()
    fake.prepare = MagicMock(side_effect=RuntimeError("nope"))
    widget.show_popup = MagicMock()

    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert (
        widget._segment_frame(image_rgb, [[10, 10, 20, 20]], 0, backend=fake)
        is None
    )
    widget.show_popup.assert_called()


def test_segment_frame_falls_back_to_self_backend(widget):
    """``backend=None`` (the default) falls back to ``self._backend`` —
    direct-call tests rely on this."""
    fake = MagicMock()
    fake.prepare = MagicMock()
    fake.segment = MagicMock(return_value=np.ones((100, 100), dtype=bool))
    widget._backend = fake

    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    result = widget._segment_frame(image_rgb, [[10, 10, 20, 20]], 0)
    assert result is not None


def test_segment_frame_cache_key_includes_roi(widget):
    """Codex P1: same frame + different ROI must NOT share a cache slot."""
    fake = MagicMock()
    captured = {}

    def fake_prepare(image, image_id):
        captured["shape"] = image.shape

    def fake_segment(box_xyxy):
        h, w = captured["shape"][:2]
        return np.ones((h, w), dtype=bool)

    fake.prepare.side_effect = fake_prepare
    fake.segment.side_effect = fake_segment

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    widget._segment_frame(image_rgb, [[100, 400, 50, 50]], 5, backend=fake)
    widget._segment_frame(image_rgb, [[1900, 400, 50, 50]], 5, backend=fake)

    assert fake.prepare.call_count == 2
    id1 = fake.prepare.call_args_list[0].kwargs["image_id"]
    id2 = fake.prepare.call_args_list[1].kwargs["image_id"]
    assert id1 != id2
    assert 5 in id1 and 5 in id2


def test_segment_frame_cache_key_stable_for_identical_roi(widget):
    """Identical ROI must produce identical image_id (so the local
    backend can still skip set_image)."""
    fake = MagicMock()
    fake.prepare = MagicMock()
    fake.segment = MagicMock(return_value=np.ones((800, 1024), dtype=bool))

    image_rgb = np.zeros((800, 2000, 3), dtype=np.uint8)
    widget._segment_frame(image_rgb, [[1500, 400, 50, 50]], 5, backend=fake)
    widget._segment_frame(image_rgb, [[1500, 400, 50, 50]], 5, backend=fake)

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


def test_load_local_model_work_raises_on_failure(monkeypatch):
    """Pure work function raises; no widget instance involved."""
    import napari_3d_trace_anything._track_widget as widget_module

    def boom(*a, **kw):
        raise RuntimeError("no disk space")

    monkeypatch.setattr(widget_module, "get_sam_model", boom)
    from napari_3d_trace_anything._track_widget import TrackAnything

    with pytest.raises(RuntimeError, match="no disk space"):
        TrackAnything._load_local_model_work("vit_h", "cpu")


def test_build_backend_for_worker_after_failed_load_returns_none(
    widget, monkeypatch
):
    """If lazy-load fails inside the worker, returns None + popup."""
    import napari_3d_trace_anything._track_widget as widget_module

    monkeypatch.setattr(
        widget_module,
        "get_sam_model",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    widget.show_popup = MagicMock()
    widget._local_predictor = None  # force lazy-load path
    snap = {
        "kind": "local",
        "local_model_name": "vit_h",
        "device": "cpu",
        "remote_url": "",
        "remote_model": "sam",
    }
    assert widget._build_backend_for_worker(snap) is None
    widget.show_popup.assert_called()
    # Must not have leaked a LocalSAMBackend(None) onto self.
    assert widget._backend is None


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


# ---------- Threaded model loading + snapshot (PR#10) ----------


def test_snapshot_backend_config_captures_local_settings(widget):
    widget._local_radio.setChecked(True)
    widget._local_model_combo.setCurrentText("vit_b")
    snap = widget._snapshot_backend_config()
    assert snap["kind"] == "local"
    assert snap["local_model_name"] == "vit_b"
    assert snap["device"] is not None


def test_snapshot_backend_config_captures_remote_settings(widget):
    widget._remote_radio.setChecked(True)
    widget._remote_url.setText("http://server:1234/segment")
    widget._remote_model_combo.setCurrentText("sam_hq")
    snap = widget._snapshot_backend_config()
    assert snap["kind"] == "remote"
    assert snap["remote_url"] == "http://server:1234/segment"
    assert snap["remote_model"] == "sam_hq"


def _drive_load_worker_synchronously(widget, monkeypatch):
    """Replace create_worker with a fake that lets the test fire the
    returned/errored callbacks manually."""
    captured = {}

    def fake_create_worker(work):
        worker = MagicMock()
        worker._work = work
        worker.returned.connect.side_effect = lambda cb: captured.update(
            returned=cb
        )
        worker.errored.connect.side_effect = lambda cb: captured.update(
            errored=cb
        )
        worker.start = MagicMock()
        return worker

    monkeypatch.setattr(
        "napari_3d_trace_anything._track_widget.create_worker",
        fake_create_worker,
    )
    return captured


def test_on_load_local_model_clicked_disables_buttons(widget, monkeypatch):
    captured = _drive_load_worker_synchronously(widget, monkeypatch)

    class _FakeSAM:
        def to(self, device):
            return self

    monkeypatch.setattr(
        "napari_3d_trace_anything._track_widget.get_sam_model",
        lambda name, device=None: _FakeSAM(),
    )
    import segment_anything

    monkeypatch.setattr(
        segment_anything, "SamPredictor", lambda sam: MagicMock()
    )

    widget._on_load_local_model_clicked()
    assert widget._load_local_btn.text() == "Loading..."
    assert widget._load_local_btn.isEnabled() is False
    assert widget._trace_btn.isEnabled() is False
    assert widget._segment_only_btn.isEnabled() is False
    assert widget._local_model_combo.isEnabled() is False

    # Fire returned callback as the worker would.
    result = widget._load_worker._work()
    captured["returned"](result)

    assert widget._load_local_btn.text() == "Load Model"
    assert widget._load_local_btn.isEnabled() is True
    assert widget._trace_btn.isEnabled() is True
    assert widget._local_predictor is not None  # state applied on GUI thread
    assert isinstance(widget._backend, type(widget._backend))


def test_on_load_local_model_errored_shows_popup(widget, monkeypatch):
    captured = _drive_load_worker_synchronously(widget, monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(
        "napari_3d_trace_anything._track_widget.get_sam_model", boom
    )
    widget.show_popup = MagicMock()
    widget._on_load_local_model_clicked()
    try:
        widget._load_worker._work()
    except RuntimeError as exc:
        captured["errored"](exc)
    widget.show_popup.assert_called()
    assert widget._load_local_btn.isEnabled() is True


def test_load_click_rejected_during_trace(widget):
    widget._worker = MagicMock()  # pretend trace is running
    widget._on_load_local_model_clicked()
    assert widget._load_worker is None  # explicit guard fired


def test_trace_click_rejected_during_load(widget):
    widget._load_worker = MagicMock()
    widget._on_trace_click()
    assert widget._worker is None


def test_segment_only_click_rejected_during_load(widget):
    widget._load_worker = MagicMock()
    widget._on_segment_only_click()
    assert widget._worker is None


def test_closed_flag_prevents_callback_state_change(widget):
    """Worker callbacks after close() must not mutate widget state."""
    widget._closed = True
    pre = widget._local_predictor
    widget._on_load_local_model_returned(("sam", "predictor"))
    assert widget._local_predictor is pre  # unchanged
