import napari
import numpy as np
from napari._qt.qthreading import create_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from ._utils import (
    SAMSegmenter,
    check_image_type,
    create_boxes_list,
    get_sam_model,
    parse_slice_range,
    preprocess,
)
from .processing.process_slice_sequence_v2 import optimize_segmentation


class TraceAnything(QWidget):
    PREDICTED_LABEL_NAME = "Predicted-Label"
    MERGED_LABEL_PREFIX = "Merged-Label"
    LABELS_PREFIX = "Labels"

    _show_popup_signal = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._labels_layer_selection = None
        self._image_type = None
        self._current_slice = None
        self._current_image_shape = None
        self._minimum_slice = 0
        self._maximum_slice = 1
        self._worker = None
        self._load_worker = None
        self._closed = False
        self._trace_params = {}
        self._pending_accept_label = None
        self._pending_accept_target = None
        self._layer_events_connected = False
        self._show_popup_signal.connect(self._show_popup_slot)

        self.vbox = QVBoxLayout()
        self._model_selection = QComboBox()
        self._model_selection.addItems(["default", "vit_h", "vit_l", "vit_b"])
        self.vbox.addWidget(self._model_selection)
        self._model_load_btn = QPushButton("load model")
        self._model_load_btn.clicked.connect(self._on_load_model_clicked)
        self.vbox.addWidget(self._model_load_btn)
        self.vbox.addWidget(QLabel("Image Layer"))
        self._image_layer_selection = QComboBox()
        self._image_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Image)
            ]
        )
        self._image_layer_selection.currentTextChanged.connect(
            self._on_image_layer_changed
        )
        self.vbox.addWidget(self._image_layer_selection)
        self.vbox.addWidget(QLabel("Labels Layer"))
        self._labels_layer_selection = QComboBox()
        self._labels_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Labels)
            ]
        )
        self.vbox.addWidget(self._labels_layer_selection)
        self.vbox.addWidget(QLabel("Slice Range"))
        slice_hbox = QHBoxLayout()
        self._start_slice = QSpinBox(
            minimum=self._minimum_slice,
            maximum=self._maximum_slice,
            value=0,
        )
        self._end_slice = QSpinBox(
            minimum=self._minimum_slice,
            maximum=self._maximum_slice,
            value=0,
        )
        slice_hbox.addWidget(self._start_slice)
        slice_hbox.addWidget(self._end_slice)
        self.vbox.addLayout(slice_hbox)
        self.vbox.addWidget(QLabel("Margin Ratio"))
        self._margin_ratio = QDoubleSpinBox()
        self._margin_ratio.setRange(-1.0, 1.0)
        self._margin_ratio.setValue(0.0)
        self._margin_ratio.setSingleStep(0.1)
        self.vbox.addWidget(self._margin_ratio)

        box_filter_hbox = QHBoxLayout()
        box_filter_left = QVBoxLayout()
        box_filter_left.addWidget(QLabel("Max Objects"))
        self._max_objects_per_label = QSpinBox()
        self._max_objects_per_label.setRange(0, 100)
        self._max_objects_per_label.setValue(0)
        self._max_objects_per_label.setToolTip(
            "Max objects per label (0 = unlimited)"
        )
        box_filter_left.addWidget(self._max_objects_per_label)
        box_filter_right = QVBoxLayout()
        box_filter_right.addWidget(QLabel("Min Area"))
        self._min_box_area = QSpinBox()
        self._min_box_area.setRange(0, 10000)
        self._min_box_area.setValue(0)
        self._min_box_area.setToolTip(
            "Min object area for box generation (0 = no filter)"
        )
        box_filter_right.addWidget(self._min_box_area)
        box_filter_hbox.addLayout(box_filter_left)
        box_filter_hbox.addLayout(box_filter_right)
        self.vbox.addLayout(box_filter_hbox)

        self.vbox.addWidget(QLabel("Self-Optimization"))
        self._self_optimization = QCheckBox()
        self.vbox.addWidget(self._self_optimization)

        morphology_hbox = QHBoxLayout()
        morphology_left = QVBoxLayout()
        morphology_left.addWidget(QLabel("Fill Holes"))
        self._hole_area_threshold = QSpinBox()
        self._hole_area_threshold.setRange(0, 1000)
        self._hole_area_threshold.setValue(0)
        self._hole_area_threshold.setToolTip(
            "Area threshold for filling holes (0 to disable)"
        )
        morphology_left.addWidget(self._hole_area_threshold)
        morphology_right = QVBoxLayout()
        morphology_right.addWidget(QLabel("Remove Obj"))
        self._min_object_size = QSpinBox()
        self._min_object_size.setRange(0, 1000)
        self._min_object_size.setValue(0)
        self._min_object_size.setToolTip(
            "Size threshold for removing objects (0 to disable)"
        )
        morphology_right.addWidget(self._min_object_size)
        morphology_hbox.addLayout(morphology_left)
        morphology_hbox.addLayout(morphology_right)
        self.vbox.addLayout(morphology_hbox)

        self.vbox.addWidget(QLabel("Instance Mode"))
        self._instance_mode = QCheckBox()
        self.vbox.addWidget(self._instance_mode)

        self._trace_btn = QPushButton("Trace")
        self._trace_btn.clicked.connect(self._trace)
        self.vbox.addWidget(self._trace_btn)
        self.vbox.addWidget(QLabel("Merged Layer"))
        self._merged_labels_layer_selection = QComboBox()
        self._merged_labels_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Labels)
            ]
        )
        self.vbox.addWidget(self._merged_labels_layer_selection)

        self.vbox.addWidget(QLabel("Accept Range (e.g., 1-3,5)"))
        self._slice_range_input = QLineEdit()
        self._slice_range_input.setPlaceholderText("1-3, 5, 9-10")
        self._slice_range_input.setToolTip(
            "Accept only specified slices (empty = all)"
        )
        self.vbox.addWidget(self._slice_range_input)

        buttons_hbox = QHBoxLayout()
        self._accept_btn = QPushButton("Accept (A)")
        self._accept_btn.clicked.connect(lambda: self._accept_prediction(None))
        buttons_hbox.addWidget(self._accept_btn)
        self._clear_btn = QPushButton("Clear (C)")
        self._clear_btn.clicked.connect(
            lambda: self._clear_current_label(None)
        )
        buttons_hbox.addWidget(self._clear_btn)
        self.vbox.addLayout(buttons_hbox)

        self._init_variables()

        self._sam_box_layer = self._viewer.add_shapes(
            name="SAM-Box",
            edge_color="red",
            edge_width=2,
            face_color="transparent",
            ndim=3,
            features=self.features,
            text=self.text,
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

        self.device = None
        self._sam_model = None
        self.sam_predictor = None
        self.sam_segmenter = None

        self._viewer.layers.events.inserted.connect(
            self._on_layer_list_changed
        )
        self._viewer.layers.events.removed.connect(self._on_layer_list_changed)
        self._layer_events_connected = True

        self._viewer.bind_key("C", self._clear_current_label)
        self._viewer.bind_key("A", self._accept_prediction)

        self._on_layer_list_changed(None)

    def _init_variables(self):
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
                    max=np.iinfo(np.uint16).max,
                )
                if ok:
                    layer.features.loc[len(layer.features) - 1, "class"] = (
                        number
                    )
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
        if event is None:
            return
        if isinstance(event.value, napari.layers.Image):
            self._image_layer_selection.clear()
            self._image_layer_selection.addItems(
                [
                    layer.name
                    for layer in self._viewer.layers
                    if isinstance(layer, napari.layers.Image)
                ]
            )
            for i, layer in enumerate(self._viewer.layers):
                if isinstance(layer, napari.layers.Image):
                    self._viewer.layers.move(i, 0)
            self._on_image_layer_changed(None)
        elif isinstance(event.value, napari.layers.Labels):
            if self._current_image_shape is not None:
                layer_name = self._image_layer_selection.currentText()
                merged_name = f"{self.MERGED_LABEL_PREFIX}-{layer_name}"
                labels_name = f"{self.LABELS_PREFIX}-{layer_name}"
                self._refresh_label_comboboxes(
                    self._current_image_shape,
                    labels_name,
                    merged_name,
                )

    def _on_load_model_clicked(self):
        """Threaded entry point — keeps the GUI responsive during DL."""
        if self._load_worker is not None or self._worker is not None:
            return  # already loading or tracing
        # Capture device on the GUI thread so the worker doesn't touch
        # torch from a possibly half-initialized state.
        if self.device is None:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        model_name = self._model_selection.currentText()
        device = self.device
        self._set_loading_state(True)
        self._load_worker = create_worker(
            lambda: self._load_model_work(model_name, device)
        )
        self._load_worker.returned.connect(self._on_load_model_returned)
        self._load_worker.errored.connect(self._on_load_model_errored)
        self._load_worker.start()

    @staticmethod
    def _load_model_work(model_name, device):
        """Pure work, no self mutation. Returns (sam, predictor)."""
        from segment_anything import SamPredictor

        sam = get_sam_model(model_name, device=device)
        return sam, SamPredictor(sam)

    def _on_load_model_returned(self, result):
        if self._closed:
            return
        sam, predictor = result
        self._sam_model = sam
        self.sam_predictor = predictor
        self.sam_segmenter = SAMSegmenter(self.sam_predictor)
        print("model loaded")
        self._set_loading_state(False)

    def _on_load_model_errored(self, exc):
        if self._closed:
            return
        self._set_loading_state(False)
        self.show_popup(f"Failed to load SAM model: {exc}")

    def _set_loading_state(self, loading):
        """Mutual exclusion: while loading, disable Trace and the model
        selector; while tracing, _set_trace_state(True) disables Load."""
        self._model_load_btn.setEnabled(not loading)
        self._model_load_btn.setText("Loading..." if loading else "load model")
        self._trace_btn.setEnabled(not loading)
        self._model_selection.setEnabled(not loading)
        if not loading:
            self._load_worker = None

    def _set_trace_state(self, active):
        """Disables Load while a Trace worker is running."""
        self._model_load_btn.setEnabled(not active)
        self._model_selection.setEnabled(not active)

    def show_popup(self, message):
        """Thread-safe popup: emits a signal so the QMessageBox is built
        on the GUI thread even when called from a worker."""
        self._show_popup_signal.emit(message)

    def _show_popup_slot(self, message):
        import sys

        print(message)
        if "pytest" in sys.modules:
            return
        msg = QMessageBox()
        msg.setWindowTitle("Trace Anything")
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        """Mark widget closed so worker callbacks don't touch destroyed UI."""
        self._closed = True
        super().closeEvent(event)

    def _suppress_layer_events(self):
        """Context-manager-like disconnect/reconnect for layer events."""
        try:
            self._viewer.layers.events.inserted.disconnect(
                self._on_layer_list_changed
            )
            self._viewer.layers.events.removed.disconnect(
                self._on_layer_list_changed
            )
            self._layer_events_connected = False
        except (TypeError, ValueError):
            pass

    def _restore_layer_events(self):
        if not self._layer_events_connected:
            self._viewer.layers.events.inserted.connect(
                self._on_layer_list_changed
            )
            self._viewer.layers.events.removed.connect(
                self._on_layer_list_changed
            )
            self._layer_events_connected = True

    def _get_or_create_labels_layer(self, name, shape):
        """Return existing layer if name matches, else create new."""
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name == name:
                return layer
        return self._viewer.add_labels(
            np.zeros(shape, dtype="uint16"),
            name=name,
            blending="additive",
            opacity=0.5,
        )

    def _refresh_label_comboboxes(
        self, image_shape, default_labels, default_merged
    ):
        """Repopulate label comboboxes with shape-compatible layers only."""
        compatible = [
            layer.name
            for layer in self._viewer.layers
            if (
                isinstance(layer, napari.layers.Labels)
                and layer.data.shape == image_shape
            )
        ]

        for combo, default in [
            (self._labels_layer_selection, default_labels),
            (self._merged_labels_layer_selection, default_merged),
        ]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(compatible)
            idx = combo.findText(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_image_layer_changed(self, index):
        layer_name = self._image_layer_selection.currentText()
        if not layer_name:
            return
        self._image_type = check_image_type(self._viewer, layer_name)
        if "stack" in self._image_type:
            self._maximum_slice = (
                self._viewer.layers[
                    self._image_layer_selection.currentText()
                ].data.shape[0]
                - 1
            )
            self._start_slice.setMaximum(self._maximum_slice)
            self._end_slice.setMaximum(self._maximum_slice)
            self._end_slice.setValue(self._maximum_slice)

            image_shape = self._viewer.layers[layer_name].data.shape
            self._current_image_shape = image_shape

            # Suppress layer events during batch layer creation
            self._suppress_layer_events()

            # Predicted-Label: recreate on shape change, clear on image switch
            self._pending_accept_label = None
            self._pending_accept_target = None

            if self._predict_label_layer is not None:
                if self._predict_label_layer.data.shape != image_shape:
                    self._viewer.layers.remove(self._predict_label_layer)
                    self._predict_label_layer = None
                else:
                    self._predict_label_layer.data = np.zeros(
                        image_shape, dtype="uint16"
                    )
                    self._predict_label_layer.refresh()
            if self._predict_label_layer is None:
                self._predict_label_layer = self._viewer.add_labels(
                    np.zeros(image_shape, dtype="uint16"),
                    name=self.PREDICTED_LABEL_NAME,
                    blending="additive",
                    opacity=0.5,
                )

            # Per-image Merged-Label and Labels layers
            merged_name = f"{self.MERGED_LABEL_PREFIX}-{layer_name}"
            labels_name = f"{self.LABELS_PREFIX}-{layer_name}"
            self._get_or_create_labels_layer(merged_name, image_shape)
            self._get_or_create_labels_layer(labels_name, image_shape)

            self._restore_layer_events()

            self._refresh_label_comboboxes(
                image_shape, labels_name, merged_name
            )

    def _trace(self):
        if self._worker:
            if self._worker.is_running:
                self._trace_btn.setText("stopping...")
                self._stop_predicting = True
                self._worker.send(self._stop_predicting)
            else:
                self._delete_worker()
            return
        # Explicit guard — independent of button enable state so tests
        # and programmatic callers can't race the load worker.
        if self._load_worker is not None:
            return
        # Read all UI values on main thread
        self._trace_params = {
            "margin_ratio": self._margin_ratio.value(),
            "self_optimization": (self._self_optimization.isChecked()),
            "instance_mode": (self._instance_mode.isChecked()),
            "hole_threshold": (self._hole_area_threshold.value()),
            "min_obj_size": (self._min_object_size.value()),
            "max_objects": (self._max_objects_per_label.value()),
            "min_area": self._min_box_area.value(),
            "image_layer": (self._image_layer_selection.currentText()),
            "labels_layer": (self._labels_layer_selection.currentText()),
            "start_slice": self._start_slice.value(),
            "end_slice": self._end_slice.value(),
        }
        self._set_trace_state(True)
        self._worker = create_worker(self._tracer)
        self._worker.started.connect(lambda: print("worker is running..."))
        self._worker.finished.connect(self._delete_worker)
        self._worker.start()
        self._stop_predicting = False
        self._trace_btn.setText("stop")

    def _delete_worker(self):
        del self._worker
        self._worker = None
        self._set_trace_state(False)
        self._trace_btn.setText("Trace")

    def _tracer(self):
        self._update_values = []
        self._pending_accept_label = None
        self._pending_accept_target = None
        image_layer = self._trace_params["image_layer"]
        labels_layer_name = self._trace_params["labels_layer"]
        start = self._trace_params["start_slice"]
        end = self._trace_params["end_slice"]
        image = self._viewer.layers[image_layer].data
        if not labels_layer_name:
            return
        if start > end:
            for i in tqdm(range(start, end - 1, -1)):
                self._predict(image, i, labels_layer_name, i + 1)
                stop_predicting = yield
                if stop_predicting:
                    break
        else:
            for i in tqdm(range(start, end + 1)):
                self._predict(image, i, labels_layer_name, i - 1)
                stop_predicting = yield
                if stop_predicting:
                    break

    def _predict(
        self, image, slice_index, labels_layer_name, prev_slice_index
    ):
        preprocessed_image = preprocess(image, self._image_type, slice_index)
        height, width = preprocessed_image.shape[:2]
        instance_mode = self._trace_params["instance_mode"]

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

        margin_ratio = self._trace_params["margin_ratio"]
        n_slices = self._predict_label_layer.data.shape[0]
        if 0 <= prev_slice_index < n_slices:
            prev_pred = self._predict_label_layer.data[prev_slice_index]
            if np.any(prev_pred > 0):
                labels = prev_pred
            else:
                labels = self._viewer.layers[labels_layer_name].data[
                    prev_slice_index
                ]
            boxes_created, label_values_created = create_boxes_list(
                labels,
                margin_ratio=margin_ratio,
                max_objects=self._trace_params["max_objects"],
                min_area=self._trace_params["min_area"],
            )
        else:
            boxes_created, label_values_created = [], []
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

        seg_kwargs = {
            "self_optimization": self._trace_params["self_optimization"],
            "margin_ratio": self._trace_params["margin_ratio"],
            "hole_threshold": self._trace_params["hole_threshold"],
            "min_obj_size": self._trace_params["min_obj_size"],
        }

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
                    y1 = max(
                        0,
                        min(height - 1024, center_y - half_size),
                    )
                    x2 = x1 + 1024
                    y2 = y1 + 1024

                    if x2 > width:
                        x2 = width
                        x1 = max(0, x2 - 1024)
                    if y2 > height:
                        y2 = height
                        y1 = max(0, y2 - 1024)

                    cropped_image = preprocessed_image[y1:y2, x1:x2]

                seg_image = cropped_image
                seg_coords = [
                    x1_box - x1,
                    y1_box - y1,
                    x2_box - x1,
                    y2_box - y1,
                ]
            else:
                seg_image = preprocessed_image
                seg_coords = coords

            mask = self._segment(seg_image, seg_coords, **seg_kwargs)

            if should_crop:
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y1:y2, x1:x2] = mask
                mask = full_mask

            if instance_mode:
                # 同じlabel_valueの初回処理時のみ古いピクセルをクリア
                if label_value not in cleared_labels:
                    current_data[current_data == label_value] = 0
                    cleared_labels.add(label_value)
                current_data[mask] = label_value
            else:
                slice_data = layer_data[slice_index]
                slice_data[mask] = 1
                layer_data[slice_index] = slice_data

        if instance_mode:
            layer_data[slice_index] = current_data

        self._predict_label_layer.data = layer_data
        self._predict_label_layer.refresh()

    def _segment(
        self,
        image,
        coords,
        self_optimization=False,
        margin_ratio=0.0,
        hole_threshold=0,
        min_obj_size=0,
    ):
        """Run segmentation with optional self-optimization
        and morphological post-processing."""
        from skimage.morphology import (
            remove_small_holes,
            remove_small_objects,
        )

        if self_optimization:
            _, mask = optimize_segmentation(
                image,
                coords,
                self.sam_segmenter,
                margin_ratio,
            )
        else:
            mask = self.sam_segmenter.segment(image, coords)

        if hole_threshold > 0:
            mask = remove_small_holes(
                mask.astype(bool),
                area_threshold=hole_threshold,
            )
        if min_obj_size > 0:
            mask = remove_small_objects(
                mask.astype(bool), min_size=min_obj_size
            )
        return mask

    def _accept_prediction(self, layer):
        if self._predict_label_layer is None:
            return
        if self._merged_labels_layer_selection.currentText() == "":
            return

        output_layer = self._viewer.layers[
            self._merged_labels_layer_selection.currentText()
        ]
        predict_data = self._predict_label_layer.data

        # Accept Range
        range_str = self._slice_range_input.text().strip()
        if range_str:
            try:
                target_slices = parse_slice_range(range_str)
                max_slice = predict_data.shape[0] - 1
                invalid = [s for s in target_slices if s < 0 or s > max_slice]
                if invalid:
                    print(f"Invalid slice numbers: {invalid}")
                    return
                if not target_slices:
                    print("No valid slices in range")
                    return
            except ValueError as e:
                print(f"Error parsing range: {e}")
                return
        else:
            target_slices = list(range(predict_data.shape[0]))

        instance_mode = self._instance_mode.isChecked()
        merged_name = self._merged_labels_layer_selection.currentText()
        if not instance_mode:
            # Reuse label from previous partial accept of the
            # same prediction into the same layer
            if (
                self._pending_accept_label is None
                or self._pending_accept_target != merged_name
            ):
                self._pending_accept_label = int(np.max(output_layer.data)) + 1
                self._pending_accept_target = merged_name
            new_label = self._pending_accept_label

        for s in target_slices:
            pred_s = predict_data[s]
            out_s = output_layer.data[s]
            transfer = (pred_s > 0) & (out_s == 0)
            if not np.any(transfer):
                continue
            if instance_mode:
                out_s[transfer] = pred_s[transfer]
            else:
                out_s[transfer] = new_label
            output_layer.data[s] = out_s

        output_layer.refresh()

        # Clear only accepted slices when range specified
        if range_str:
            for s in target_slices:
                self._predict_label_layer.data[s] = 0
        else:
            self._predict_label_layer.data = np.zeros_like(predict_data)

        # Reset pending label when prediction layer is fully clear
        if not np.any(self._predict_label_layer.data > 0):
            self._pending_accept_label = None
            self._pending_accept_target = None

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
