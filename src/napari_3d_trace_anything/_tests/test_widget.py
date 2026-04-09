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
