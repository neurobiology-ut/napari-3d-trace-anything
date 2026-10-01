from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_3d_trace_anything import TraceAnything


@pytest.fixture
def widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((10, 100, 100)), name="test-image")
    w = TraceAnything(viewer)
    return w


def merged_layer(widget):
    return widget._viewer.layers[
        f"{TraceAnything.MERGED_LABEL_PREFIX}-test-image"
    ]


def test_widget_creation(widget):
    assert widget is not None
    assert widget._instance_mode is not None
    assert not widget._instance_mode.isChecked()


def test_instance_mode_checkbox(widget):
    assert not widget._instance_mode.isChecked()
    widget._instance_mode.setChecked(True)
    assert widget._instance_mode.isChecked()
    widget._instance_mode.setChecked(False)
    assert not widget._instance_mode.isChecked()


def test_accept_prediction_instance_mode(widget):
    widget._instance_mode.setChecked(True)

    predict_data = widget._predict_label_layer.data.copy()
    predict_data[0, 10:20, 10:20] = 2
    predict_data[0, 50:60, 50:60] = 3
    widget._predict_label_layer.data = predict_data

    widget._accept_prediction(None)

    ml = merged_layer(widget)
    assert np.any(ml.data[0] == 2)
    assert np.any(ml.data[0] == 3)
    assert np.all(widget._predict_label_layer.data == 0)


def test_accept_prediction_non_instance_mode(widget):
    widget._instance_mode.setChecked(False)

    predict_data = widget._predict_label_layer.data.copy()
    predict_data[0, 10:20, 10:20] = 1
    widget._predict_label_layer.data = predict_data

    widget._accept_prediction(None)

    ml = merged_layer(widget)
    assert np.any(ml.data[0] == 1)
    assert np.all(widget._predict_label_layer.data == 0)


def test_accept_prediction_instance_mode_no_overwrite(widget):
    widget._instance_mode.setChecked(True)

    ml = merged_layer(widget)
    merged_data = ml.data.copy()
    merged_data[0, 10:20, 10:20] = 5
    ml.data = merged_data

    predict_data = widget._predict_label_layer.data.copy()
    predict_data[0, 10:20, 10:20] = 2
    widget._predict_label_layer.data = predict_data

    widget._accept_prediction(None)

    assert np.all(ml.data[0, 10:20, 10:20] == 5)


def _trace_params(labels_name, instance_mode=False):
    return {
        "instance_mode": instance_mode,
        "margin_ratio": 0.0,
        "max_objects": 0,
        "min_area": 0,
        "self_optimization": False,
        "hole_threshold": 0,
        "min_obj_size": 0,
        "image_layer": "test-image",
        "labels_layer": labels_name,
        "start_slice": 0,
        "end_slice": 0,
    }


def test_predict_non_instance_mode_overwrites(widget, monkeypatch):
    """Non-instance mode write should overwrite to 1, not accumulate to >=2."""
    labels_name = f"{TraceAnything.LABELS_PREFIX}-test-image"

    # Auto-propagate seed on slice 0 (so create_boxes_list yields one box)
    labels_layer = widget._viewer.layers[labels_name]
    data = labels_layer.data.copy()
    data[0, 10:30, 10:30] = 1
    labels_layer.data = data

    # Stale value 5 on slice 1: must become 1, not 6, after the write
    predict_data = widget._predict_label_layer.data.copy()
    predict_data[1, 10:30, 10:30] = 5
    widget._predict_label_layer.data = predict_data

    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    monkeypatch.setattr(widget, "_segment", lambda *a, **kw: mask)
    widget.sam_segmenter = object()

    widget._trace_params = _trace_params(labels_name)

    image = widget._viewer.layers["test-image"].data
    widget._predict(image, 1, labels_name, 0)

    unique_vals = set(np.unique(widget._predict_label_layer.data[1]).tolist())
    assert unique_vals == {0, 1}, f"got {unique_vals}"


def test_predict_skips_out_of_range_prev_slice(widget, monkeypatch):
    """prev_slice_index outside [0, n_slices) must skip auto-box generation
    and must not raise IndexError at the boundaries."""
    n_slices = widget._predict_label_layer.data.shape[0]
    labels_name = f"{TraceAnything.LABELS_PREFIX}-test-image"

    # Put labels on the last slice — they must NOT be used as seed for slice 0
    labels_layer = widget._viewer.layers[labels_name]
    data = labels_layer.data.copy()
    data[n_slices - 1, 10:30, 10:30] = 1
    labels_layer.data = data

    segment_calls = []

    def fake_segment(*args, **kwargs):
        segment_calls.append(args)
        return np.zeros((100, 100), dtype=bool)

    monkeypatch.setattr(widget, "_segment", fake_segment)
    widget.sam_segmenter = object()

    widget._trace_params = _trace_params(labels_name)

    image = widget._viewer.layers["test-image"].data

    # Forward boundary: slice 0, prev=-1
    widget._predict(image, 0, labels_name, -1)
    assert (
        segment_calls == []
    ), "No SAM-Box and prev out of range — _segment must not be called"

    # Backward boundary: last slice, prev=n_slices — must not raise
    widget._predict(image, n_slices - 1, labels_name, n_slices)


# ---------- Threaded model loading (PR#10) ----------


def test_load_button_disables_during_worker(widget, monkeypatch):
    """Clicking Load Model puts the UI in `Loading...` until the worker
    returns; on returned the button comes back to normal."""

    class _FakeSAM:
        def to(self, device):
            return self

    monkeypatch.setattr(
        "napari_3d_trace_anything._widget.get_sam_model",
        lambda name, device=None: _FakeSAM(),
    )
    # SamPredictor constructor inspects sam.image_encoder; bypass it.
    import segment_anything

    monkeypatch.setattr(
        segment_anything, "SamPredictor", lambda sam: MagicMock()
    )
    # Force "cpu" so the worker doesn't probe torch device availability.
    widget.device = "cpu"

    captured = {}

    def fake_create_worker(work):
        # Drive the worker synchronously: capture the work() result and
        # the returned/errored callbacks so the test can fire them.
        worker = MagicMock()
        worker._work = work

        def connect_returned(cb):
            captured["returned"] = cb

        def connect_errored(cb):
            captured["errored"] = cb

        worker.returned.connect.side_effect = connect_returned
        worker.errored.connect.side_effect = connect_errored
        worker.start = MagicMock()
        return worker

    monkeypatch.setattr(
        "napari_3d_trace_anything._widget.create_worker",
        fake_create_worker,
    )

    widget._on_load_model_clicked()
    assert widget._model_load_btn.text() == "Loading..."
    assert widget._model_load_btn.isEnabled() is False
    assert widget._trace_btn.isEnabled() is False

    # Now simulate worker completion on the GUI thread.
    result = widget._load_worker._work()
    captured["returned"](result)

    assert widget._model_load_btn.text() == "load model"
    assert widget._model_load_btn.isEnabled() is True
    assert widget._trace_btn.isEnabled() is True
    assert widget.sam_segmenter is not None


def test_load_failure_shows_popup(widget, monkeypatch):
    """If the worker raises, errored callback shows popup and restores UI."""
    widget.device = "cpu"

    def boom(*a, **kw):
        raise RuntimeError("no disk space")

    monkeypatch.setattr("napari_3d_trace_anything._widget.get_sam_model", boom)
    widget.show_popup = MagicMock()

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
        "napari_3d_trace_anything._widget.create_worker",
        fake_create_worker,
    )

    widget._on_load_model_clicked()
    try:
        widget._load_worker._work()
    except RuntimeError as exc:
        captured["errored"](exc)

    widget.show_popup.assert_called()
    assert widget._model_load_btn.isEnabled() is True


def test_load_click_rejected_during_trace(widget):
    """Trace in progress → Load click is a no-op (explicit guard)."""
    widget._worker = MagicMock()  # pretend trace is running
    widget._on_load_model_clicked()
    # No new load worker was created.
    assert widget._load_worker is None


def test_trace_click_rejected_during_load(widget):
    """Load in progress → Trace click is a no-op (explicit guard)."""
    widget._load_worker = MagicMock()  # pretend load is running
    # _trace is the toggler — when load is active, it must early-return.
    widget._trace()
    assert widget._worker is None


def test_closed_flag_prevents_callback_state_change(widget):
    """Worker callbacks fired after widget.close() must NOT mutate state."""
    widget._closed = True
    # Build a sentinel result that, if applied, would set sam_segmenter.
    before = widget.sam_segmenter
    widget._on_load_model_returned(("sentinel-sam", "sentinel-predictor"))
    assert widget.sam_segmenter is before  # unchanged


def test_failed_load_clears_sam_state(widget):
    """Errored callback drops any previously-loaded segmenter so a
    subsequent Trace can't silently reuse the stale state."""
    widget._sam_model = object()
    widget.sam_predictor = MagicMock()
    widget.sam_segmenter = MagicMock()
    widget.show_popup = MagicMock()

    widget._on_load_model_errored(RuntimeError("boom"))

    assert widget._sam_model is None
    assert widget.sam_predictor is None
    assert widget.sam_segmenter is None
    widget.show_popup.assert_called()
