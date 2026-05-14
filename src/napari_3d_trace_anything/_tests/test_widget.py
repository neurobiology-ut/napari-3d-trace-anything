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

    unique_vals = set(
        np.unique(widget._predict_label_layer.data[1]).tolist()
    )
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
    assert segment_calls == [], (
        "No SAM-Box and prev out of range — _segment must not be called"
    )

    # Backward boundary: last slice, prev=n_slices — must not raise
    widget._predict(image, n_slices - 1, labels_name, n_slices)
