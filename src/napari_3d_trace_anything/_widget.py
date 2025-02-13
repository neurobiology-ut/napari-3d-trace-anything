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
)
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import label
from skimage.transform import resize
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

        self._sam_box_layer = self._viewer.add_shapes(
            name="SAM-Box",
            edge_color="red",
            edge_width=2,
            face_color="transparent",
            ndim=3,
        )
        self.lock_controls(self._sam_box_layer)

        if self._image_layer_selection.currentText() != "":
            self._image_type = check_image_type(
                self._viewer, self._image_layer_selection.currentText()
            )
            if "stack" in self._image_type:
                print("image type check passed")
                self._on_image_layer_changed(None)
                # add predict-label layer
                self._predict_label_layer = self._viewer.add_labels(
                    np.zeros(
                        self._viewer.layers[
                            self._image_layer_selection.currentText()
                        ].data.shape,
                        dtype="uint16",
                    ),
                    name="Predicted-Label",
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
                self._merged_label_layer = self._viewer.add_labels(
                    np.zeros(
                        self._viewer.layers[
                            self._image_layer_selection.currentText()
                        ].data.shape,
                        dtype="uint16",
                    ),
                    name="Merged-Label",
                    blending="additive",
                    opacity=0.5,
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

            else:
                print("image type check failed")
                print("image must be stack")

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
        print("image_layer_changed")
        self._image_type = check_image_type(
            self._viewer, self._image_layer_selection.currentText()
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
        print("start tracing")
        image = self._viewer.layers[
            self._image_layer_selection.currentText()
        ].data
        print("target_image_shape: ", image.shape)
        print("start slice: ", self._start_slice.value())
        print("end slice: ", self._end_slice.value())
        print(self._sam_box_layer.data)
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

    def _predict(self, image, i, labels_layer_name, prev_slice):
        preprocessed_image = preprocess(image, self._image_type, i)
        height, width = preprocessed_image.shape[:2]
        boxes = [x for x in self._sam_box_layer.data if x[0][0] == i]
        if len(boxes) == 0:
            labels = label(
                self._viewer.layers[labels_layer_name].data[prev_slice] > 0
                )
            margin_ratio = self._margin_ratio.value()
            boxes = create_boxes_list(labels, margin_ratio=margin_ratio)
        else:
            labels = None

        for idx, coords in enumerate(boxes):
            if self.sam_segmenter is not None:
                # クロップが必要かどうかを判断
                should_crop = width > 1024 or height > 1024

                if should_crop:
                    # ボックスの座標を取得 (y, x)
                    # coordsは[z, y, x]形式
                    x_coords = [c[2] for c in coords]  # x座標を取得
                    y_coords = [c[1] for c in coords]  # y座標を取得
                    
                    # バウンディングボックスの座標を計算
                    x1_box = min(x_coords)
                    x2_box = max(x_coords)
                    y1_box = min(y_coords)
                    y2_box = max(y_coords)

                    print(f"box (x, y): ({x1_box}, {y1_box}) - ({x2_box}, {y2_box})")
                    
                    # ボックスの中心座標を計算
                    center_x = int((x1_box + x2_box) / 2)
                    center_y = int((y1_box + y2_box) / 2)

                    # クロップ範囲を計算（1024x1024を確保）
                    half_size = 512
                    x1 = max(0, min(width - 1024, center_x - half_size))
                    y1 = max(0, min(height - 1024, center_y - half_size))
                    x2 = x1 + 1024
                    y2 = y1 + 1024

                    # 画像の端に到達した場合の調整
                    if x2 > width:
                        x2 = width
                        x1 = max(0, x2 - 1024)
                    if y2 > height:
                        y2 = height
                        y1 = max(0, y2 - 1024)

                    print(f"crop (x, y): ({x1}, {y1}) - ({x2}, {y2})")

                    # クロップされた画像を作成
                    cropped_image = preprocessed_image[y1:y2, x1:x2]
                    # 1024x1024にpaddingをする
                    cropped_image = np.pad(
                        cropped_image,
                        [
                            (0, 1024 - cropped_image.shape[0]),
                            (0, 1024 - cropped_image.shape[1]),
                        ],
                        mode="constant",
                    )
                    # labelsもクロップ
                    if labels is not None:
                        cropped_labels = labels[y1:y2, x1:x2]
                        cropped_labels = np.pad(
                            cropped_labels,
                            [
                                (0, 1024 - cropped_labels.shape[0]),
                                (0, 1024 - cropped_labels.shape[1]),
                            ],
                            mode="constant",
                        )
                        # skimageで256x256にリサイズ
                        cropped_labels = resize(
                            cropped_labels,
                            (256, 256),
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        ).astype(np.uint8)
                        mask = cropped_labels == (idx + 1)
                        # (C, H, W)に変換
                        mask = mask[None, :, :]

                    else:
                        cropped_labels = None
                        mask = None

                    # ボックス座標をクロップ後の座標系に変換
                    cropped_coords = [
                        x1_box - x1,  # x1
                        y1_box - y1,  # y1
                        x2_box - x1,  # x2
                        y2_box - y1   # y2
                    ]

                    print(f"coords after crop (x, y): {cropped_coords}")

                    if self._self_optimization.isChecked():
                        # optimize_segmentationを使用
                        _, cropped_mask = optimize_segmentation(
                            cropped_image,
                            cropped_coords,
                            self.sam_segmenter,
                            self._margin_ratio.value(),
                            mask_input=mask
                        )
                    else:
                        cropped_mask = self.sam_segmenter.segment(
                            cropped_image, cropped_coords, mask_input=mask)

                    # マスクをオリジナルサイズに戻す
                    mask = np.zeros((height, width), dtype=bool)
                    mask[y1:y2, x1:x2] = cropped_mask

                else:
                    if self._self_optimization.isChecked():
                        # optimize_segmentationを使用
                        _, mask = optimize_segmentation(
                            preprocessed_image,
                            coords,
                            self.sam_segmenter,
                            self._margin_ratio.value()
                        )
                    else:
                        mask = self.sam_segmenter.segment(
                            preprocessed_image, coords)
                print("add mask to labels layer")
                viewer_layer = self._viewer.layers[labels_layer_name]
                layer_data = viewer_layer.data
                layer_data[i] = layer_data[i] + mask * 1
                viewer_layer.data = layer_data
                self._viewer.layers[labels_layer_name].refresh()
            else:
                print("model not loaded")

    def _accept_prediction(self, layer):
        if (
            self._labels_layer_selection.currentText() != ""
            and self._merged_labels_layer_selection.currentText() != ""
        ):
            print("start label-transfer")
            input_layer = self._viewer.layers[
                self._labels_layer_selection.currentText()
            ]
            output_layer = self._viewer.layers[
                self._merged_labels_layer_selection.currentText()
            ]
            if isinstance(input_layer, napari.layers.labels.labels.Labels):
                max_output_layer_label = np.max(output_layer.data).astype(
                    np.uint16
                )
                output_layer.data += (
                    (input_layer.data == 1).astype(np.uint8)
                    * (output_layer.data == 0).astype(np.uint8)
                ).astype(np.uint16) * (max_output_layer_label + 1)
                self._predict_label_layer.data = np.zeros_like(
                    self._predict_label_layer.data
                )
            print("finish label-transfer")
        else:
            print("not accepted")
            pass

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
