import napari
import numpy as np
import torch
from napari._qt.qthreading import create_worker
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QComboBox, QLabel, QSpinBox
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

from ._utils import check_image_type, load_model, preprocess, create_box, change_image_dtype


class TraceAnything(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._labels_layer_selection = None
        self._image_type = None
        self._current_slice = None
        self._minimum_slice = 0
        self._maximum_slice = 1
        self._worker = None

        self.vbox = QVBoxLayout()
        self._model_selection = QComboBox()
        self._model_selection.addItems(list(sam_model_registry.keys()))
        self.vbox.addWidget(self._model_selection)
        self._model_load_btn = QPushButton("load model")
        self._model_load_btn.clicked.connect(self._load_model)
        self.vbox.addWidget(self._model_load_btn)
        self.vbox.addWidget(QLabel("input image layer"))
        self._image_layer_selection = QComboBox()
        self._image_layer_selection.addItems(
            [layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)])
        self._image_layer_selection.currentTextChanged.connect(self._on_image_layer_changed)
        self.vbox.addWidget(self._image_layer_selection)
        self.vbox.addWidget(QLabel("output labels layer"))
        self._labels_layer_selection = QComboBox()
        self._labels_layer_selection.addItems(
            [layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.labels.labels.Labels)])
        self.vbox.addWidget(self._labels_layer_selection)
        self.vbox.addWidget(QLabel("start slice"))
        self._start_slice = QSpinBox(minimum=self._minimum_slice, maximum=self._maximum_slice, value=0)
        self.vbox.addWidget(self._start_slice)
        self.vbox.addWidget(QLabel("end slice"))
        self._end_slice = QSpinBox(minimum=self._minimum_slice, maximum=self._maximum_slice, value=0)
        self.vbox.addWidget(self._end_slice)
        self._trace_btn = QPushButton("trace")
        self._trace_btn.clicked.connect(self._trace)
        self.vbox.addWidget(self._trace_btn)
        # add predict-merge layer selection
        self.vbox.addWidget(QLabel("merged labels layer"))
        self._merged_labels_layer_selection = QComboBox()
        self._merged_labels_layer_selection.addItems(
            [layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.labels.labels.Labels)])
        self._merged_labels_layer_selection.currentTextChanged.connect(self._on_image_layer_changed)
        self.vbox.addWidget(self._merged_labels_layer_selection)

        self._sam_box_layer = self._viewer.add_shapes(name="SAM-Box", edge_color="red", edge_width=2,
                                                      face_color="transparent", ndim=3)
        self.lock_controls(self._sam_box_layer)

        if self._image_layer_selection.currentText() != "":
            self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
            if "stack" in self._image_type:
                print("image type check passed")
                print("THE VERSION IS FRAP")
                self._on_image_layer_changed(None)
                # add predict-label layer
                self._predict_label_layer = self._viewer.add_labels(
                    np.zeros(self._viewer.layers[self._image_layer_selection.currentText()].data.shape, dtype="uint16"), 
                    name="Predicted-Label", blending="additive", opacity=0.5)
                self._labels_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.labels.labels.Labels)])
                self._merged_label_layer = self._viewer.add_labels(
                    np.zeros(self._viewer.layers[self._image_layer_selection.currentText()].data.shape, dtype="uint16"), 
                    name="Merged-Label", blending="translucent", opacity=0.5)
                self._merged_labels_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.labels.labels.Labels)])
                               
            else:
                print("image type check failed")
                print("image must be stack")

        self.setLayout(self.vbox)
        self.show()

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self._sam_model = None
        self.sam_predictor = None

        self._viewer.layers.events.inserted.connect(self._on_layer_list_changed)
        self._viewer.layers.events.removed.connect(self._on_layer_list_changed)

        self._viewer.bind_key("C", self._clear_current_label)
        self._viewer.bind_key("A", self._accept_prediction)

        self._on_layer_list_changed(None)

    def _clear_current_label(self, event):
        self._current_slice, _, _ = self._viewer.dims.current_step
        if self._viewer.layers[self._labels_layer_selection.currentText()].data[self._current_slice].sum() > 0:
            self._viewer.layers[self._labels_layer_selection.currentText()].data[self._current_slice] = 0
            self._viewer.layers[self._labels_layer_selection.currentText()].refresh()

    def _on_layer_list_changed(self, event):
        if event is not None:
            print(event.value)
            if isinstance(event.value, napari.layers.image.image.Image):
                self._image_layer_selection.clear()
                self._image_layer_selection.addItems(
                    [layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)])
                [self._viewer.layers.move(i, 0) for i, layer in enumerate(self._viewer.layers) if
                 isinstance(layer, napari.layers.image.image.Image)]
                self._on_image_layer_changed(None)
            elif isinstance(event.value, napari.layers.labels.labels.Labels):
                self._labels_layer_selection.clear()
                self._labels_layer_selection.addItems([layer.name for layer in self._viewer.layers if (isinstance(layer, napari.layers.labels.labels.Labels))])
            else:
                pass

    def _load_model(self):
        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        print("model loaded")

    def _on_image_layer_changed(self, index):
        print("image_layer_changed")
        self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
        if "stack" in self._image_type:
            self._maximum_slice = self._viewer.layers[self._image_layer_selection.currentText()].data.shape[0] - 1
            self._start_slice.setMaximum(self._maximum_slice)
            self._end_slice.setMaximum(self._maximum_slice)

    def _trace(self):
        if self._worker:
            if self._worker.is_running:
                self._trace_btn.setText('stopping...')
                self._stop_predicting = True
                self._worker.send(self._stop_predicting)
            else:
                self.delete_worker()
        else:
            self._worker = create_worker(self._tracer)
            self._worker.started.connect(lambda: print("worker is running..."))
            self._worker.finished.connect(self._delete_worker)
            self._worker.start()
            self._stop_predicting = False
            self._trace_btn.setText('stop')

    def _delete_worker(self):
        del self._worker
        self._worker = None
        self._trace_btn.setText('trace')

    def _tracer(self):
        print("start tracing")
        image = self._viewer.layers[self._image_layer_selection.currentText()].data
        print("target_image_shape: ", image.shape)
        print("start slice: ", self._start_slice.value())
        print("end slice: ", self._end_slice.value())
        print(self._sam_box_layer.data)
        labels_layer_name = self._labels_layer_selection.currentText()
        if self._start_slice.value() > self._end_slice.value():
            for i in tqdm(range(self._start_slice.value(), self._end_slice.value() - 1, -1)):
                self._predict(image, i, labels_layer_name, i + 1)
                stop_predicting = yield
                if stop_predicting:
                    break
        else:
            for i in tqdm(range(self._start_slice.value(), self._end_slice.value() + 1)):
                self._predict(image, i, labels_layer_name, i - 1)
                stop_predicting = yield
                if stop_predicting:
                    break

    def _predict(self, image, i, labels_layer_name, prev_slice):
        image = change_image_dtype(image)
        self.sam_predictor.set_image(preprocess(image, self._image_type, i))
        boxes = [x for x in self._sam_box_layer.data if x[0][0] == i]
        if len(boxes) == 0:
            boxes = create_box(self._viewer.layers[labels_layer_name].data[prev_slice])
        for coords in boxes:
            buffer = 3
            y1 = max(int(coords[0][1]) + buffer, 0)
            x1 = max(int(coords[0][2]) + buffer, 0)
            y2 = min(int(coords[2][1]) - buffer, image.shape[1])
            x2 = min(int(coords[2][2]) - buffer, image.shape[2])
            print(x1, y1, x2, y2)
            input_box = np.array([x1, y1, x2, y2])
            if self.sam_predictor is not None:
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                self._viewer.layers[labels_layer_name].data[i] = self._viewer.layers[labels_layer_name].data[i] + masks[
                    0] * 1
                self._viewer.layers[labels_layer_name].refresh()
            else:
                print("model not loaded")


    def _accept_prediction(self, layer):
        if self._labels_layer_selection.currentText() != "" and self._merged_labels_layer_selection.currentText() != "":
            print("start label-transfer")
            input_layer = self._viewer.layers[self._labels_layer_selection.currentText()]
            output_layer = self._viewer.layers[self._merged_labels_layer_selection.currentText()]
            if isinstance(input_layer, napari.layers.labels.labels.Labels):
                max_output_layer_label = np.max(output_layer.data).astype(np.uint16)
                output_layer.data += ((input_layer.data==1).astype(np.uint8)*(output_layer.data==0).astype(np.uint8)).astype(np.uint16) * (max_output_layer_label+1)
                self._predict_label_layer.data = np.zeros_like(self._predict_label_layer.data)
            print("finish label-transfer")
        else:
            print("not accepted")
            pass


    def lock_controls(self, layer, locked=True):
        widget_list = [
            'ellipse_button',
            'line_button',
            'path_button',
            'vertex_remove_button',
            'vertex_insert_button',
            'move_back_button',
            'move_front_button',
            'polygon_button',
        ]
        qctrl = self._viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widget_list:
            getattr(qctrl, wdg).setEnabled(not locked)

    def print_corner_value(self):
        print(self._viewer.dims.current_step)
        print(self._viewer.layers[self._image_layer_selection.currentText()].corner_pixels)


