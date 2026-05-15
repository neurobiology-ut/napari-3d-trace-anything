# Time-series tracking widget adapted from napari-gc-analysis (Apache-2.0)
# https://github.com/neurobiology-ut/napari-gc-analysis

import contextlib

import napari
import numpy as np
from napari._qt.qthreading import create_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.color import gray2rgb

from . import tracking
from ._utils import get_sam_model
from .sam_backends import LocalSAMBackend, RemoteSAMBackend


class TrackAnything(QWidget):
    """Time-series object tracking widget.

    Draw a single bounding box on one frame, then VitTracker propagates
    it across subsequent frames while SAM segments each tracked box.
    Frame-skip detection (ECC / POC / AKAZE / Tracker) lets the loop
    bridge gaps where the object is briefly occluded or out-of-frame.
    """

    BOX_LAYER_PREFIX = "Track-Box"
    LABEL_LAYER_PREFIX = "Track-Label"
    LOCAL_MODELS = ("default", "vit_h", "vit_l", "vit_b")
    REMOTE_MODELS = ("sam", "sam_hq")
    SKIP_METHODS = ("ECC", "POC", "AKAZE", "Tracker")

    _show_popup_signal = Signal(str)

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        self._worker = None
        self._show_popup_signal.connect(self._show_popup_slot)

        self._backend: object | None = None
        self._local_predictor = None
        self._local_sam_model = None
        self._device = None
        self._box_layer = None
        self._label_layer = None
        self._image_layer_name: str | None = None
        # Set by _on_image_layer_changed: 'gray' (T,H,W),
        # 'gray_ch' (T,H,W,1), or 'rgb' (T,H,W,3).
        self._image_kind: str | None = None
        self._frame_stack_shape: tuple | None = None
        # Selected layer tracked by object identity so a rename does not
        # require name-based lookup (which would break for the renamed
        # layer and silently steal the selection from a non-renamed one).
        self._selected_layer = None
        # Worker-scoped snapshots captured at _trace / _segment_only
        # start. The UI thread can change _image_layer_name / _label_layer
        # mid-run (rename, combo change) and the worker must keep writing
        # into the layers it started with.
        self._active_image_layer = None
        self._active_label_layer = None
        self._active_image_id: str | None = None
        # Per-Image-layer events.name handlers we hooked up — needed for
        # disconnect on layer removal / widget close.
        self._name_handlers: dict[int, object] = {}

        self._build_ui()
        self._connect_layer_events()
        self._refresh_image_combo()
        self._on_image_layer_changed()

    # ---------------- UI ----------------

    def _build_ui(self):
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel("Image Layer"))
        self._image_combo = QComboBox()
        self._image_combo.currentTextChanged.connect(
            lambda _name: self._on_image_layer_changed()
        )
        self.layout().addWidget(self._image_combo)

        # --- SAM backend group ---
        backend_group = QGroupBox("SAM Backend")
        backend_layout = QVBoxLayout()

        radio_row = QHBoxLayout()
        self._local_radio = QRadioButton("Local")
        self._remote_radio = QRadioButton("Remote")
        self._local_radio.setChecked(True)
        self._backend_buttons = QButtonGroup(self)
        self._backend_buttons.addButton(self._local_radio)
        self._backend_buttons.addButton(self._remote_radio)
        self._local_radio.toggled.connect(self._on_backend_toggled)
        radio_row.addWidget(self._local_radio)
        radio_row.addWidget(self._remote_radio)
        backend_layout.addLayout(radio_row)

        # Local sub-section
        self._local_box = QGroupBox("Local")
        local_layout = QVBoxLayout()
        local_model_row = QHBoxLayout()
        local_model_row.addWidget(QLabel("Model:"))
        self._local_model_combo = QComboBox()
        self._local_model_combo.addItems(self.LOCAL_MODELS)
        local_model_row.addWidget(self._local_model_combo)
        local_layout.addLayout(local_model_row)
        self._load_local_btn = QPushButton("Load Model")
        self._load_local_btn.clicked.connect(self._load_local_model)
        local_layout.addWidget(self._load_local_btn)
        self._local_box.setLayout(local_layout)
        backend_layout.addWidget(self._local_box)

        # Remote sub-section
        self._remote_box = QGroupBox("Remote")
        remote_layout = QVBoxLayout()
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self._remote_url = QLineEdit()
        self._remote_url.setPlaceholderText("http://host:port/segment")
        self._remote_url.setToolTip("Segmentation server URL")
        url_row.addWidget(self._remote_url)
        remote_layout.addLayout(url_row)
        remote_model_row = QHBoxLayout()
        remote_model_row.addWidget(QLabel("Model:"))
        self._remote_model_combo = QComboBox()
        self._remote_model_combo.addItems(self.REMOTE_MODELS)
        remote_model_row.addWidget(self._remote_model_combo)
        remote_layout.addLayout(remote_model_row)
        self._remote_box.setLayout(remote_layout)
        backend_layout.addWidget(self._remote_box)

        backend_group.setLayout(backend_layout)
        self.layout().addWidget(backend_group)
        self._on_backend_toggled()

        # --- Frame skip group ---
        skip_group = QGroupBox("Frame Skip Settings")
        skip_group.setCheckable(True)
        skip_group.setChecked(False)
        skip_layout = QVBoxLayout()

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._skip_method = QComboBox()
        self._skip_method.addItems(self.SKIP_METHODS)
        self._skip_method.setToolTip(
            "ECC: Enhanced Correlation Coefficient\n"
            "POC: Phase-Only Correlation\n"
            "AKAZE: Feature-based matching\n"
            "Tracker: VitTracker prediction"
        )
        method_row.addWidget(self._skip_method)
        skip_layout.addLayout(method_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Movement threshold:"))
        self._skip_threshold = QDoubleSpinBox()
        self._skip_threshold.setRange(0.1, 1000.0)
        self._skip_threshold.setValue(10.0)
        self._skip_threshold.setSingleStep(0.5)
        threshold_row.addWidget(self._skip_threshold)
        skip_layout.addLayout(threshold_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max frames to skip:"))
        self._skip_max_frames = QSpinBox()
        self._skip_max_frames.setRange(1, 1000)
        self._skip_max_frames.setValue(10)
        self._skip_max_frames.setToolTip(
            "Maximum consecutive frames that may be skipped before "
            "tracking stops."
        )
        max_row.addWidget(self._skip_max_frames)
        skip_layout.addLayout(max_row)

        skip_group.setLayout(skip_layout)
        self._skip_group = skip_group
        self.layout().addWidget(skip_group)

        # --- Segment only group ---
        segment_group = QGroupBox("Segment Only Mode")
        segment_group.setCheckable(True)
        segment_group.setChecked(False)
        segment_layout = QVBoxLayout()
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From:"))
        self._segment_from = QSpinBox()
        self._segment_from.setRange(0, 10000)
        range_row.addWidget(self._segment_from)
        range_row.addWidget(QLabel("To:"))
        self._segment_to = QSpinBox()
        self._segment_to.setRange(0, 10000)
        self._segment_to.setValue(100)
        range_row.addWidget(self._segment_to)
        segment_layout.addLayout(range_row)
        self._segment_only_btn = QPushButton("Segment Only")
        self._segment_only_btn.clicked.connect(self._on_segment_only_click)
        segment_layout.addWidget(self._segment_only_btn)
        segment_group.setLayout(segment_layout)
        self._segment_group = segment_group
        self.layout().addWidget(segment_group)

        # --- Misc ---
        self._debug_mode = QCheckBox("Debug mode (skip segmentation)")
        self.layout().addWidget(self._debug_mode)

        buttons_row = QHBoxLayout()
        self._trace_btn = QPushButton("Trace on!")
        self._trace_btn.clicked.connect(self._on_trace_click)
        buttons_row.addWidget(self._trace_btn)
        self._stop_btn = QPushButton("STOP!")
        self._stop_btn.clicked.connect(self._stop)
        buttons_row.addWidget(self._stop_btn)
        self._reset_btn = QPushButton("Reset box")
        self._reset_btn.clicked.connect(self._reset_box)
        buttons_row.addWidget(self._reset_btn)
        self.layout().addLayout(buttons_row)

    def _on_backend_toggled(self):
        is_local = self._local_radio.isChecked()
        self._local_box.setEnabled(is_local)
        self._remote_box.setEnabled(not is_local)
        # Drop the cached backend so the next trace rebuilds it from the
        # active radio's settings without carrying over a stale `prepare`.
        self._backend = None

    # ---------------- Layer events ----------------

    def _connect_layer_events(self):
        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
        # Subscribe to rename on every existing Image layer so the combo
        # follows the rename without going stale.
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._subscribe_layer_name(layer)

    def _subscribe_layer_name(self, layer):
        def handler(_event):
            self._refresh_image_combo()

        layer.events.name.connect(handler)
        self._name_handlers[id(layer)] = (layer, handler)

    def _unsubscribe_layer_name(self, layer):
        entry = self._name_handlers.pop(id(layer), None)
        if entry is None:
            return
        _layer, handler = entry
        with contextlib.suppress(TypeError, ValueError, RuntimeError):
            layer.events.name.disconnect(handler)

    def _on_layer_inserted(self, event):
        if isinstance(event.value, napari.layers.Image):
            self._subscribe_layer_name(event.value)
        self._refresh_image_combo()

    def _on_layer_removed(self, event):
        if isinstance(event.value, napari.layers.Image):
            self._unsubscribe_layer_name(event.value)
        # If the removed layer was the selected one, clear the reference.
        if event.value is self._selected_layer:
            self._selected_layer = None
        self._refresh_image_combo()

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        """Disconnect every event we subscribed to so widget teardown
        doesn't leave orphaned callbacks hanging on layer instances or
        on the viewer's layer list."""
        for layer, handler in list(self._name_handlers.values()):
            with contextlib.suppress(TypeError, ValueError, RuntimeError):
                layer.events.name.disconnect(handler)
        self._name_handlers.clear()
        with contextlib.suppress(TypeError, ValueError, RuntimeError):
            self._viewer.layers.events.inserted.disconnect(
                self._on_layer_inserted
            )
        with contextlib.suppress(TypeError, ValueError, RuntimeError):
            self._viewer.layers.events.removed.disconnect(
                self._on_layer_removed
            )
        super().closeEvent(event)

    @staticmethod
    def _classify_image_layer(layer):
        """Return (kind, frame_stack_shape) or (None, None) if unsupported.

        The tracker only handles time-series stacks. A single 2D frame
        has no time axis to propagate through; RGB without a time axis
        (e.g. (H, W, 3)) is rejected for the same reason.

        - (T, H, W) where last dim != 3  → 'gray'
        - (T, H, W, 1)                   → 'gray_ch'
        - (T, H, W, 3)                   → 'rgb'
        - else                           → unsupported

        ``frame_stack_shape`` is (T, H, W) for any of the supported
        kinds, used to size the label layer regardless of channel
        suffix.

        Edge cases:
        - ``(T, H, 3)``: rejected. We cannot distinguish a width-3
          grayscale stack from a single RGB frame, so bias toward
          rejecting the ambiguous case.
        - ``(3, H, W)`` where W != 3: accepted as ``'gray'`` — a
          3-frame grayscale stack whose width is not 3.
        - ``(T, H, W, 3)``: always accepted as ``'rgb'``, regardless
          of how small H or W happen to be.
        """
        s = layer.data.shape
        n = len(s)
        if n == 3 and s[-1] != 3:
            return "gray", s
        if n == 4 and s[-1] == 1:
            return "gray_ch", s[:3]
        if n == 4 and s[-1] == 3:
            return "rgb", s[:3]
        return None, None

    def _supported_image_layers(self):
        for layer in self._viewer.layers:
            if not isinstance(layer, napari.layers.Image):
                continue
            kind, _ = self._classify_image_layer(layer)
            if kind is not None:
                yield layer

    def _refresh_image_combo(self):
        """Rebuild combo from supported Image layers.

        Selection is restored by **object identity** against
        ``self._selected_layer`` — this is the only way to follow a
        renamed layer (its name changed in-place) while NOT being
        misled into following an unrelated layer's rename.
        """
        previous_text = self._image_combo.currentText()
        supported = list(self._supported_image_layers())
        names = [layer.name for layer in supported]
        self._image_combo.blockSignals(True)
        self._image_combo.clear()
        self._image_combo.addItems(names)
        # Restore by identity. ``in`` would compare by ``__eq__``, which
        # napari's Layer overrides — use ``is`` explicitly.
        restored_name = None
        if self._selected_layer is not None:
            for layer in supported:
                if layer is self._selected_layer:
                    restored_name = layer.name
                    break
        if restored_name is not None:
            self._image_combo.setCurrentText(restored_name)
        self._image_combo.blockSignals(False)
        new_text = self._image_combo.currentText()
        if new_text != previous_text:
            self._on_image_layer_changed()
        else:
            self._trace_btn.setEnabled(bool(new_text))

    def _on_image_layer_changed(self):
        name = self._image_combo.currentText()
        self._image_layer_name = name or None
        if not name:
            self._image_kind = None
            self._frame_stack_shape = None
            self._selected_layer = None
            self._trace_btn.setEnabled(False)
            return

        image_layer = self._viewer.layers[name]
        kind, frame_stack_shape = self._classify_image_layer(image_layer)
        if kind is None:
            # Defense in depth: _refresh_image_combo() filters unsupported
            # layers, but tests / programmatic combo manipulation can
            # still reach this path.
            self.show_popup(
                f"Image layer {name!r} shape {image_layer.data.shape} is not "
                "a supported time-series. Expected (T, H, W), (T, H, W, 1), "
                "or (T, H, W, 3)."
            )
            self._image_kind = None
            self._frame_stack_shape = None
            self._selected_layer = None
            self._trace_btn.setEnabled(False)
            return

        self._image_kind = kind
        self._frame_stack_shape = frame_stack_shape
        self._selected_layer = image_layer
        self._trace_btn.setEnabled(True)

        box_name = f"{self.BOX_LAYER_PREFIX}-{name}"
        label_name = f"{self.LABEL_LAYER_PREFIX}-{name}"

        self._box_layer = self._get_or_create_box_layer(box_name)
        # Label layer is always (T, H, W) regardless of channel suffix.
        self._label_layer = self._get_or_create_label_layer(
            label_name, frame_stack_shape
        )

        # Sync segment-only spinbox range to the image's frame count.
        T = frame_stack_shape[0]
        max_idx = max(0, T - 1)
        for spin in (self._segment_from, self._segment_to):
            spin.setMaximum(max_idx)
        # Clamp current values (setMaximum does not auto-clamp).
        self._segment_to.setValue(min(self._segment_to.value(), max_idx))
        self._segment_from.setValue(min(self._segment_from.value(), max_idx))
        # Preserve from <= to.
        if self._segment_from.value() > self._segment_to.value():
            self._segment_from.setValue(self._segment_to.value())

    def _get_or_create_box_layer(self, name):
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == name:
                return layer
        return self._viewer.add_shapes(
            name=name,
            edge_color="green",
            edge_width=2,
            face_color="transparent",
            ndim=3,
        )

    def _get_or_create_label_layer(self, name, shape):
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name == name:
                if layer.data.shape != shape:
                    layer.data = np.zeros(shape, dtype="uint8")
                return layer
        return self._viewer.add_labels(
            np.zeros(shape, dtype="uint8"),
            name=name,
            opacity=0.5,
        )

    # ---------------- Backend ----------------

    def _pick_device(self):
        if self._device is not None:
            return self._device
        import torch

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        return self._device

    def _load_local_model(self):
        """Load SAM weights + build predictor. Returns True on success.

        On failure, clears predictor / backend state so a follow-up
        ``_ensure_backend()`` doesn't wrap None in a ``LocalSAMBackend``
        and crash on the next ``segment()`` call.

        TODO(PR#10): this runs synchronously on the GUI thread; the
        first vit_h download can take minutes and there is no progress
        feedback. Threading is deferred to a follow-up PR that also
        touches TraceAnything's matching code path.
        """
        from segment_anything import SamPredictor

        device = self._pick_device()
        model_name = self._local_model_combo.currentText()
        try:
            sam = get_sam_model(model_name, device=device)
            predictor = SamPredictor(sam)
        except Exception as exc:  # noqa: BLE001 - many failure paths
            self._local_sam_model = None
            self._local_predictor = None
            self._backend = None
            self.show_popup(f"Failed to load local SAM model: {exc}")
            return False

        self._local_sam_model = sam
        self._local_predictor = predictor
        self._backend = LocalSAMBackend(self._local_predictor)
        print(f"Local SAM model {model_name!r} loaded on {device}")
        return True

    def _ensure_backend(self):
        if self._local_radio.isChecked():
            if self._local_predictor is None and not self._load_local_model():
                return None  # popup already shown
            if not isinstance(self._backend, LocalSAMBackend):
                self._backend = LocalSAMBackend(self._local_predictor)
            return self._backend

        url = self._remote_url.text().strip()
        if not url:
            self.show_popup("Remote URL is empty")
            return None
        model = self._remote_model_combo.currentText()
        if (
            not isinstance(self._backend, RemoteSAMBackend)
            or self._backend.url != url
            or self._backend.model != model
        ):
            self._backend = RemoteSAMBackend(url=url, model=model)
        return self._backend

    # ---------------- Tracing ----------------

    def _on_trace_click(self):
        if self._worker is not None:
            return
        self._worker = create_worker(self._trace)
        self._worker.started.connect(lambda: print("tracing..."))
        self._worker.yielded.connect(self._update_label_layer)
        self._worker.finished.connect(self._delete_worker)
        self._worker.start()

    def _on_segment_only_click(self):
        if self._worker is not None:
            return
        self._worker = create_worker(self._segment_only)
        self._worker.started.connect(lambda: print("segmenting..."))
        self._worker.yielded.connect(self._update_label_layer)
        self._worker.finished.connect(self._delete_worker)
        self._worker.start()

    def _update_label_layer(self, value):
        mask, index = value
        # Prefer the worker-scoped snapshots so a mid-run rename or
        # combo change can't redirect the mask write into the wrong
        # label layer or call refresh() with a stale name.
        label_layer = (
            self._active_label_layer
            if self._active_label_layer is not None
            else self._label_layer
        )
        image_layer = self._active_image_layer
        if image_layer is None and self._image_layer_name is not None:
            image_layer = self._viewer.layers[self._image_layer_name]
        if label_layer is None or image_layer is None:
            return
        label_layer.data[index] = mask
        label_layer.refresh()
        self._viewer.dims.set_current_step(0, index)
        image_layer.refresh()

    def _stop(self):
        if self._worker is not None:
            with contextlib.suppress(StopIteration, RuntimeError):
                self._worker.send(True)
            self._delete_worker()

    def _delete_worker(self):
        self._worker = None
        self._active_image_layer = None
        self._active_label_layer = None
        self._active_image_id = None
        print("worker stopped")

    def _reset_box(self):
        if self._box_layer is not None:
            self._box_layer.data = []
            self._box_layer.refresh()

    # ---- core loops ----

    def _trace(self):
        if self._image_layer_name is None or self._image_kind is None:
            self.show_popup("Please select a supported image layer.")
            return

        # Capture worker-scoped snapshots BEFORE _ensure_backend() —
        # model load can take seconds during which the UI thread could
        # change combo selection or layers could be renamed.
        image_layer = self._viewer.layers[self._image_layer_name]
        label_layer = self._label_layer
        frame_stack_shape = self._frame_stack_shape
        self._active_image_layer = image_layer
        self._active_label_layer = label_layer
        self._active_image_id = self._image_layer_name

        backend = self._ensure_backend()
        if backend is None:
            return

        index = self._viewer.dims.current_step[0]
        images = image_layer.data
        if label_layer.data.shape != frame_stack_shape:
            label_layer.data = np.zeros(frame_stack_shape, dtype="uint8")

        coords = self._get_bounding_box(index)
        if coords is None:
            return

        skip_enabled = self._skip_group.isChecked()
        skip_method = self._skip_method.currentText().lower()
        skip_threshold = self._skip_threshold.value()
        skip_max_frames = self._skip_max_frames.value()

        tracker = tracking.get_vit_tracker()
        # Reused across skip checks when skip_method == 'tracker' so the
        # ONNX is not re-read for every checked frame. Lazy because
        # creation is unnecessary for ECC / POC / AKAZE methods.
        skip_tracker = None
        ref_image = self._as_rgb(self._extract_frame(images, index))

        n_frames = frame_stack_shape[0]
        i = index
        while i < n_frames:
            image = self._as_rgb(self._extract_frame(images, i))

            if i == index:
                tracker.init(image, tuple(coords[0]))
            else:
                if skip_enabled:
                    if skip_method == "tracker" and skip_tracker is None:
                        skip_tracker = tracking.get_vit_tracker()
                    new_i = self._advance_through_skips(
                        images,
                        ref_image,
                        coords[0],
                        i,
                        skip_threshold,
                        skip_max_frames,
                        skip_method,
                        skip_tracker=skip_tracker,
                    )
                    if new_i is None:
                        return
                    if new_i != i:
                        # Re-init the existing tracker (do not re-read
                        # ONNX from disk) — TrackerVit.init() supports
                        # re-initialization on the same instance.
                        i = new_i
                        image = self._as_rgb(self._extract_frame(images, i))
                        tracker.init(ref_image, tuple(coords[0]))

                ok, bbox = tracker.update(image)
                if ok:
                    coords = [[int(v) for v in bbox]]
                    self._set_shapes_data(coords, i)
                else:
                    self.show_popup("Tracking failed")
                    return
                ref_image = image

            if self._debug_mode.isChecked():
                self._viewer.dims.set_current_step(0, i)
                image_layer.refresh()
            else:
                result = self._segment_frame(image, coords, i)
                if result is None:
                    i += 1
                    continue
                stop_sign = yield result
                if stop_sign is True:
                    break

            i += 1

    def _segment_only(self):
        """Run segmentation on existing boxes only (no tracking).

        The ``Debug mode (skip segmentation)`` checkbox is intentionally
        ignored here: Segment Only's whole purpose is to run segmentation
        on pre-existing boxes, so skipping it would be a no-op.
        """
        if self._image_layer_name is None or self._image_kind is None:
            self.show_popup("Please select a supported image layer.")
            return

        # Capture active state before model load (see _trace).
        image_layer = self._viewer.layers[self._image_layer_name]
        label_layer = self._label_layer
        frame_stack_shape = self._frame_stack_shape
        self._active_image_layer = image_layer
        self._active_label_layer = label_layer
        self._active_image_id = self._image_layer_name

        backend = self._ensure_backend()
        if backend is None:
            return

        images = image_layer.data
        if label_layer.data.shape != frame_stack_shape:
            label_layer.data = np.zeros(frame_stack_shape, dtype="uint8")

        n_frames = frame_stack_shape[0]
        frame_from = self._segment_from.value()
        frame_to = min(self._segment_to.value(), n_frames - 1)
        if frame_from > frame_to:
            self.show_popup("Invalid frame range: 'From' must be <= 'To'")
            return

        processed = skipped = 0
        for i in range(frame_from, frame_to + 1):
            coords = self._get_bounding_box(i, silent=True)
            if coords is None:
                skipped += 1
                continue
            image = self._as_rgb(self._extract_frame(images, i))
            result = self._segment_frame(image, coords, i)
            if result is None:
                continue
            processed += 1
            stop_sign = yield result
            if stop_sign is True:
                break
        print(
            f"Segment only: {processed} processed, {skipped} skipped (no box)"
        )

    def _advance_through_skips(
        self,
        images,
        ref_image,
        bbox_xywh,
        start_idx,
        threshold,
        max_frames,
        method,
        *,
        skip_tracker=None,
    ):
        """Walk forward from start_idx while movement exceeds threshold.

        Returns the first index where movement is acceptable, or None if
        more than ``max_frames`` consecutive frames had to be skipped.
        ``max_frames=N`` means *up to N frames* may be skipped before
        we bail; this changed from the previous off-by-one semantics
        where ``N`` actually allowed ``N-1`` skips (PR#9 review L602).

        Performance note (tracker method): when ``method == 'tracker'``
        and skip detection is enabled, the temp tracker inside
        ``check_frame_movement`` runs an init+update per checked frame,
        and when this function returns the outer ``_trace`` loop then
        calls ``tracker.update`` on the same image again — producing a
        second inference per non-skipped frame. The temp tracker's bbox
        isn't reused because the main tracker carries incremental
        state from prior frames; a cleaner fix requires splitting
        skip-detection and main-loop tracker state more explicitly.
        Tracked as a known perf cost — see CLAUDE.md "Known
        limitations".
        """
        skip_count = 0
        cur = start_idx
        n_frames = self._frame_stack_shape[0]
        while cur < n_frames:
            check_image = self._as_rgb(self._extract_frame(images, cur))
            exceeds, dist, _dx, _dy, _new = tracking.check_frame_movement(
                ref_image,
                check_image,
                bbox_xywh,
                threshold=threshold,
                method=method,
                tracker=skip_tracker,
            )
            if not exceeds:
                if skip_count > 0:
                    print(
                        f"Resuming at frame {cur} after skipping "
                        f"{skip_count} frames (movement: {dist:.2f}px)"
                    )
                return cur
            skip_count += 1
            print(
                f"Frame {cur}: movement {dist:.2f}px exceeds threshold "
                f"({method}), skipping"
            )
            if skip_count > max_frames:
                self.show_popup(
                    f"Skipped {max_frames} consecutive frames. "
                    "Tracking stopped."
                )
                return None
            cur += 1
        self.show_popup("Reached end of frames while skipping")
        return None

    # ---- segmentation ----

    def _segment_frame(self, image_rgb, boxes_xywh, frame_index):
        """Crop large images, run SAM, paste mask back to full size.

        Returns (mask, frame_index) or None on backend error.
        """
        backend = self._ensure_backend()
        if backend is None:
            return None

        img_h, img_w = image_rgb.shape[:2]
        needs_crop, x1, y1, x2, y2 = tracking.compute_roi_for_segmentation(
            image_rgb.shape, boxes_xywh
        )

        # Stable ID across mid-run rename. ``_active_image_id`` is set
        # by _trace / _segment_only at start; for direct calls (tests)
        # it may be None, in which case fall back to the live name.
        layer_id = (
            self._active_image_id
            if self._active_image_id is not None
            else self._image_layer_name
        )

        if needs_crop:
            crop = image_rgb[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
            # ROI must be part of the cache key: rerunning on the same
            # frame with a different box position can produce a different
            # crop, and the backend must NOT short-circuit prepare() on
            # the stale image.
            image_id = (layer_id, frame_index, x1, y1, x2, y2)
        else:
            crop = image_rgb
            offset_x = offset_y = 0
            image_id = (layer_id, frame_index, None)

        try:
            backend.prepare(crop, image_id=image_id)
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            self.show_popup(f"backend.prepare failed: {exc}")
            return None

        full_mask = np.zeros((img_h, img_w), dtype=bool)
        for box in boxes_xywh:
            bx, by, bw, bh = (int(v) for v in box)
            box_xyxy = np.array(
                [
                    bx - offset_x,
                    by - offset_y,
                    bx + bw - offset_x,
                    by + bh - offset_y,
                ],
                dtype=np.float32,
            )
            # noqa: BLE001 below — backend failures span requests, pickle,
            # cv2, torch; we surface them all to the user as a popup.
            try:
                mask = backend.segment(box_xyxy)
            except Exception as exc:  # noqa: BLE001
                self.show_popup(f"backend.segment failed: {exc}")
                return None
            if needs_crop:
                full_mask[y1:y2, x1:x2] |= mask
            else:
                full_mask |= mask
        return full_mask, frame_index

    # ---------------- helpers ----------------

    @staticmethod
    def _as_rgb(image):
        return gray2rgb(image) if image.ndim == 2 else image

    def _extract_frame(self, images, index):
        """Return frame ``index`` as a 2D gray or (H, W, 3) RGB array.

        Channel suffix (T, H, W, 1) is squeezed so the rest of the
        pipeline can assume frames are either 2D or RGB.
        """
        frame = images[index]
        if self._image_kind == "gray_ch":
            return frame[..., 0]
        return frame

    def _get_bounding_box(self, index, silent=False):
        """Return [[x, y, width, height]] for the box on ``index``.

        Returns None when no box exists for that frame, or when multiple
        boxes are drawn (the tracker only supports one object at a time).
        """
        if self._box_layer is None:
            if not silent:
                self.show_popup("No box layer")
            return None
        boxes = [x for x in self._box_layer.data if x[0][0] == index]
        output = []
        for coords in boxes:
            ys = [int(c[1]) for c in coords]
            xs = [int(c[2]) for c in coords]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            output.append([x1, y1, x2 - x1, y2 - y1])
        if not output:
            if not silent:
                self.show_popup("No box found")
            return None
        if len(output) > 1:
            if not silent:
                self.show_popup("Multiple boxes found")
            return None
        return output

    def _set_shapes_data(self, coords, z):
        if self._box_layer is None:
            return
        self._box_layer.data = [
            x for x in self._box_layer.data if x[0][0] != z
        ]
        rectangles = []
        for coord in coords:
            x1, y1, width, height = (int(v) for v in coord)
            x2, y2 = x1 + width, y1 + height
            rectangles.append(
                np.array(
                    [
                        [z, y1, x1],
                        [z, y1, x2],
                        [z, y2, x2],
                        [z, y2, x1],
                    ]
                )
            )
        self._box_layer.add_rectangles(rectangles)
        self._box_layer.refresh()

    def show_popup(self, message):
        """Thread-safe popup. Emits signal so the message box is shown on
        the GUI thread, even when called from the napari worker."""
        self._show_popup_signal.emit(message)

    def _show_popup_slot(self, message):
        import sys

        print(message)
        if "pytest" in sys.modules:
            return
        msg = QMessageBox()
        msg.setWindowTitle("Track Anything")
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
