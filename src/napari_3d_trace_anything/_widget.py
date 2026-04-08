import napari
import numpy as np
import torch
from napari._qt.qthreading import create_worker
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QWidget,
    QComboBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QInputDialog,
)
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from ._utils import (
    check_image_type,
    load_model,
    preprocess,
    create_boxes_list,
    SAMSegmenter,
)
from .processing.process_slice_sequence_v2 import optimize_segmentation


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
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.image.image.Image)
            ]
        )
        self._image_layer_selection.currentTextChanged.connect(
            self._on_image_layer_changed
        )
        self.vbox.addWidget(self._image_layer_selection)
        self.vbox.addWidget(QLabel("output labels layer"))
        self._labels_layer_selection = QComboBox()
        self._labels_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.labels.labels.Labels)
            ]
        )
        self.vbox.addWidget(self._labels_layer_selection)
        self.vbox.addWidget(QLabel("start slice"))
        self._start_slice = QSpinBox(
            minimum=self._minimum_slice, maximum=self._maximum_slice, value=0
        )
        self.vbox.addWidget(self._start_slice)
        self.vbox.addWidget(QLabel("end slice"))
        self._end_slice = QSpinBox(
            minimum=self._minimum_slice, maximum=self._maximum_slice, value=0
        )
        self.vbox.addWidget(self._end_slice)
        # add margin ratio input (optional)
        self.vbox.addWidget(QLabel("margin ratio (optional)"))
        self._margin_ratio = QDoubleSpinBox()
        self._margin_ratio.setRange(-1.0, 1.0)
        self._margin_ratio.setValue(0.0)
        self._margin_ratio.setSingleStep(0.1)
        self.vbox.addWidget(self._margin_ratio)

        # self-optimizationのチェックボックスを追加
        self.vbox.addWidget(QLabel("self-optimization"))
        self._self_optimization = QCheckBox()
        self.vbox.addWidget(self._self_optimization)

        # instance mode のチェックボックスを追加
        self.vbox.addWidget(QLabel("instance mode"))
        self._instance_mode = QCheckBox()
        self.vbox.addWidget(self._instance_mode)

        self._trace_btn = QPushButton("trace")
        self._trace_btn.clicked.connect(self._trace)
        self.vbox.addWidget(self._trace_btn)
        # add predict-merge layer selection
        self.vbox.addWidget(QLabel("merged labels layer"))
        self._merged_labels_layer_selection = QComboBox()
        self._merged_labels_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.labels.labels.Labels)
            ]
        )
        self._merged_labels_layer_selection.currentTextChanged.connect(
            self._on_image_layer_changed
        )
        self.vbox.addWidget(self._merged_labels_layer_selection)

        self.initVariables()

        self._sam_box_layer = self._viewer.add_shapes(
            name="SAM-Box",
            edge_color="red",
            edge_width=2,
            face_color="transparent",
            ndim=3,
            features=self.features,
            text=self.text
        )
        self._sam_box_layer.features = self._sam_box_layer.features.astype(
            {"class": int}
        )
        self._sam_box_layer.feature_defaults["class"] = 1
        self._sam_boxes = self._sam_box_layer.data
        self._sam_box_layer.mouse_drag_callbacks.append(self.popup)
        self.lock_controls(self._sam_box_layer)

        self._predict_label_layer = None

        if self._image_layer_selection.currentText() != "":
            self._on_image_layer_changed(None)

        self.setLayout(self.vbox)
        self.show()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._sam_model = None
        self.sam_predictor = None

        self._viewer.layers.events.inserted.connect(
            self._on_layer_list_changed
        )
        self._viewer.layers.events.removed.connect(self._on_layer_list_changed)

        self._viewer.bind_key("C", self._clear_current_label)
        self._viewer.bind_key("A", self._accept_prediction)

        self._on_layer_list_changed(None)

    def initVariables(self):
        """Initializes the variables."""
        self.features = {"class": []}
        self.text = {
            "string": "{class}",
            "anchor": "upper_left",
            "translation": [0, 0, 0],
            "size": 12,
            "color": "green",
        }
        self._update_values = []

    def popup(self, layer, event):
        """Popup for SAM-Box layer"""
        # mouse click
        yield
        # mouse move
        while event.type == "mouse_move":
            yield
        # mouse release
        if self._sam_box_layer.mode == "add_rectangle":
            if self._instance_mode.isChecked():
                number, ok = QInputDialog.getInt(
                    self,
                    "Numbering",
                    "Enter instance number:",
                    value=1,
                    min=1,
                )
                if ok:
                    layer.features.loc[
                        len(layer.features) - 1, "class"
                    ] = number
                    layer.refresh_text()
                else:
                    # キャンセル時はboxを削除
                    layer.data = layer.data[:-1]

    def _clear_current_label(self, event):
        self._current_slice, _, _ = self._viewer.dims.current_step
        if (
            self._viewer.layers[self._labels_layer_selection.currentText()]
            .data[self._current_slice]
            .sum()
            > 0
        ):
            self._viewer.layers[
                self._labels_layer_selection.currentText()
            ].data[self._current_slice] = 0
            self._viewer.layers[
                self._labels_layer_selection.currentText()
            ].refresh()

    def _on_layer_list_changed(self, event):
        if event is not None:
            print(event.value)
            if isinstance(event.value, napari.layers.image.image.Image):
                self._image_layer_selection.clear()
                self._image_layer_selection.addItems(
                    [
                        layer.name
                        for layer in self._viewer.layers
                        if isinstance(layer, napari.layers.image.image.Image)
                    ]
                )
                [
                    self._viewer.layers.move(i, 0)
                    for i, layer in enumerate(self._viewer.layers)
                    if isinstance(layer, napari.layers.image.image.Image)
                ]
                self._on_image_layer_changed(None)
            elif isinstance(event.value, napari.layers.labels.labels.Labels):
                self._labels_layer_selection.clear()
                self._labels_layer_selection.addItems(
                    [
                        layer.name
                        for layer in self._viewer.layers
                        if isinstance(
                            layer,
                            napari.layers.labels.labels.Labels
                        )
                    ]
                )
            else:
                pass

    def _load_model(self):
        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        # SAMSegmenterインスタンスの作成
        self.sam_segmenter = SAMSegmenter(self.sam_predictor)
        print("model loaded")

    def _on_image_layer_changed(self, index):
        layer_name = self._image_layer_selection.currentText()
        if not layer_name:
            return
        self._image_type = check_image_type(
            self._viewer, layer_name
        )
        if "stack" in self._image_type:
            self._maximum_slice = (
                self._viewer.layers[
                    self._image_layer_selection.currentText()
                ].data.shape[0]
                - 1
            )
            self._start_slice.setMaximum(self._maximum_slice)
            self._end_slice.setMaximum(self._maximum_slice)

            image_shape = self._viewer.layers[
                layer_name
            ].data.shape

            # Create Predicted-Label layer if not yet created
            if self._predict_label_layer is None:
                self._predict_label_layer = self._viewer.add_labels(
                    np.zeros(image_shape, dtype="uint16"),
                    name="Predicted-Label",
                    blending="additive",
                    opacity=0.5,
                )
                self._viewer.add_labels(
                    np.zeros(image_shape, dtype="uint16"),
                    name="Merged-Label",
                    blending="additive",
                    opacity=0.5,
                )
                self._labels_layer_selection.addItems(
                    [
                        layer.name
                        for layer in self._viewer.layers
                        if isinstance(
                            layer, napari.layers.labels.labels.Labels
                        )
                    ]
                )
                self._merged_labels_layer_selection.addItems(
                    [
                        layer.name
                        for layer in self._viewer.layers
                        if isinstance(
                            layer, napari.layers.labels.labels.Labels
                        )
                    ]
                )

    def _trace(self):
        if self._worker:
            if self._worker.is_running:
                self._trace_btn.setText("stopping...")
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
            self._trace_btn.setText("stop")

    def _delete_worker(self):
        del self._worker
        self._worker = None
        self._trace_btn.setText("trace")

    def _tracer(self):
        self._update_values = []
        image = self._viewer.layers[
            self._image_layer_selection.currentText()
        ].data
        labels_layer_name = self._labels_layer_selection.currentText()
        if self._start_slice.value() > self._end_slice.value():
            for i in tqdm(
                range(
                    self._start_slice.value(), self._end_slice.value() - 1, -1
                )
            ):
                self._predict(image, i, labels_layer_name, i + 1)
                stop_predicting = yield
                if stop_predicting:
                    break
        else:
            for i in tqdm(
                range(self._start_slice.value(), self._end_slice.value() + 1)
            ):
                self._predict(image, i, labels_layer_name, i - 1)
                stop_predicting = yield
                if stop_predicting:
                    break

    def _predict(
        self, image, slice_index, labels_layer_name, prev_slice_index
    ):
        preprocessed_image = preprocess(image, self._image_type, slice_index)
        height, width = preprocessed_image.shape[:2]
        instance_mode = self._instance_mode.isChecked()

        boxes = []
        label_values = []
        for x, label_value in zip(
            self._sam_box_layer.data,
            self._sam_box_layer.features["class"],
        ):
            if x[0][0] == slice_index:
                boxes.append(x)
                label_value = int(label_value)
                label_values.append(label_value)
                if label_value not in self._update_values:
                    self._update_values.append(label_value)

        labels = self._viewer.layers[labels_layer_name].data[prev_slice_index]
        margin_ratio = self._margin_ratio.value()

        boxes_created, label_values_created = create_boxes_list(
            labels, margin_ratio=margin_ratio
        )
        # 手動boxがあるラベルは自動生成boxを使わない
        manual_label_set = set(label_values)
        for box, label_value in zip(boxes_created, label_values_created):
            if label_value not in manual_label_set:
                boxes.append(box)
                label_values.append(label_value)

        if self.sam_segmenter is None or not boxes:
            return

        cropped_image = None
        x1 = y1 = x2 = y2 = 0
        layer_data = self._predict_label_layer.data
        should_crop = width > 1024 or height > 1024

        if instance_mode:
            # label_value ごとに最初のbox処理前に一度だけクリアするため、
            # 処理済みlabel_valueを追跡する
            current_data = layer_data[slice_index].astype(np.int32)
            cleared_labels = set()

        for coords, label_value in zip(boxes, label_values):
            if should_crop:
                x_coords = [c[2] for c in coords]
                y_coords = [c[1] for c in coords]

                x1_box = min(x_coords)
                x2_box = max(x_coords)
                y1_box = min(y_coords)
                y2_box = max(y_coords)

                x_margin = x2_box - x1_box
                y_margin = y2_box - y1_box

                if cropped_image is not None:
                    if not (
                        x1_box > x1 + x_margin
                        and x2_box < x2 - x_margin
                        and y1_box > y1 + y_margin
                        and y2_box < y2 - y_margin
                    ):
                        cropped_image = None

                if cropped_image is None:
                    center_x = int((x1_box + x2_box) / 2)
                    center_y = int((y1_box + y2_box) / 2)

                    half_size = 512
                    x1 = max(0, min(width - 1024, center_x - half_size))
                    y1 = max(0, min(height - 1024, center_y - half_size))
                    x2 = x1 + 1024
                    y2 = y1 + 1024

                    if x2 > width:
                        x2 = width
                        x1 = max(0, x2 - 1024)
                    if y2 > height:
                        y2 = height
                        y1 = max(0, y2 - 1024)

                    cropped_image = preprocessed_image[y1:y2, x1:x2]

                cropped_coords = [
                    x1_box - x1,
                    y1_box - y1,
                    x2_box - x1,
                    y2_box - y1,
                ]

                mask = self._segment(cropped_image, cropped_coords)
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y1:y2, x1:x2] = mask
                mask = full_mask

            else:
                mask = self._segment(preprocessed_image, coords)

            if instance_mode:
                # 同じlabel_valueの初回処理時のみ古いピクセルをクリア
                if label_value not in cleared_labels:
                    current_data[current_data == label_value] = 0
                    cleared_labels.add(label_value)
                current_data[mask] = label_value
            else:
                layer_data[slice_index] = (
                    layer_data[slice_index] + mask * 1
                )

        if instance_mode:
            layer_data[slice_index] = current_data

        self._predict_label_layer.data = layer_data
        self._predict_label_layer.refresh()

    def _segment(self, image, coords):
        """Run segmentation with optional self-optimization."""
        if self._self_optimization.isChecked():
            _, mask = optimize_segmentation(
                image,
                coords,
                self.sam_segmenter,
                self._margin_ratio.value(),
            )
        else:
            mask = self.sam_segmenter.segment(image, coords)
        return mask

    def _accept_prediction(self, layer):
        if (
            self._merged_labels_layer_selection.currentText() == ""
        ):
            return

        output_layer = self._viewer.layers[
            self._merged_labels_layer_selection.currentText()
        ]
        predict_data = self._predict_label_layer.data

        transfer_mask = (predict_data > 0) & (output_layer.data == 0)
        if self._instance_mode.isChecked():
            output_layer.data[transfer_mask] = predict_data[
                transfer_mask
            ]
        else:
            max_label = int(np.max(output_layer.data))
            output_layer.data[transfer_mask] = max_label + 1

        output_layer.refresh()
        self._predict_label_layer.data = np.zeros_like(
            predict_data
        )
        self._predict_label_layer.refresh()

    def lock_controls(self, layer, locked=True):
        widget_list = [
            "ellipse_button",
            "line_button",
            "path_button",
            "vertex_remove_button",
            "vertex_insert_button",
            "move_back_button",
            "move_front_button",
            "polygon_button",
        ]
        qctrl = self._viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widget_list:
            getattr(qctrl, wdg).setEnabled(not locked)

    def print_corner_value(self):
        print(self._viewer.dims.current_step)
        print(
            self._viewer.layers[
                self._image_layer_selection.currentText()
            ].corner_pixels
        )
