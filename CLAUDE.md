# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

napari-3d-trace-anything is a napari plugin for 3D object tracing using Meta's Segment Anything Model (SAM). Users draw bounding boxes on one slice and the plugin propagates segmentation across subsequent slices automatically, with optional iterative refinement.

## Development Setup

```bash
pip install -e ".[testing]"
```

## Common Commands

```bash
# Run all tests
pytest -v --cov=napari_3d_trace_anything --cov-report=xml

# Run a single test file
pytest src/napari_3d_trace_anything/_tests/test_clamp.py -v

# Run with tox (matrix: py38/39/310)
tox

# Lint and format
black src/napari_3d_trace_anything
ruff check src/napari_3d_trace_anything --fix

# Pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Plugin Entry Point

Napari discovers the plugin via the NPE2 manifest system:
- `pyproject.toml` `napari.manifest` entry point → `src/napari_3d_trace_anything/napari.yaml` → two widgets:
  - `_widget.py:TraceAnything` — z-stack propagation
  - `_track_widget.py:TrackAnything` — time-series tracking + Local/Remote SAM

### Widgets

| Widget | Use case | Box layer | Output layer(s) |
|---|---|---|---|
| `TraceAnything` | 3D z-stack: propagate labels slice→slice | `SAM-Box` (single, shared) | `Predicted-Label`, `Labels-{img}`, `Merged-Label-{img}` |
| `TrackAnything` | Time-series: track + segment one object across frames | `Track-Box-{img}` (per image) | `Track-Label-{img}` (per image) |

The two widgets are independent. Layer names are deliberately disjoint so both can run side-by-side without clobbering each other; cross-feature wiring (e.g. seeding `TrackAnything` from `TraceAnything` output) is not implemented.

### Core Modules

- **`_widget.py`** — `TraceAnything(QWidget)`: z-stack propagation widget. Manages UI controls (model selection, slice range, margin ratio, self-optimization, instance mode), a `SAM-Box` shapes layer for manual box placement, and a threaded worker (`_tracer` generator) for batch slice processing. The `_predict` method is the core segmentation loop: it generates boxes from the previous slice's labels, runs SAM, and handles large images by cropping a 1024×1024 ROI around the active box.

- **`_track_widget.py`** — `TrackAnything(QWidget)`: time-series tracking widget adapted from `napari-gc-analysis`. One box is drawn on a starting frame; OpenCV `TrackerVit` propagates it forward and SAM segments each tracked box. Supports Local/Remote SAM via UI toggle, optional frame-skip detection (ECC / POC / AKAZE / Tracker) for occlusions, and a Segment Only mode that re-runs SAM on pre-existing boxes.

- **`tracking.py`** — Pure functions adapted from `napari-gc-analysis`: `get_vit_tracker()` (loads bundled ONNX), displacement estimators (`compute_ecc_displacement`, `compute_poc_displacement`, `compute_akaze_displacement`, `compute_tracker_displacement`), `extract_roi_around_box`, `check_frame_movement` (dispatcher), and `compute_roi_for_segmentation` (centers a ≤1024px crop on the boxes, expanding when their union spans more). Used by `TrackAnything`.

- **`sam_backends.py`** — `SAMBackend` Protocol with two-stage API (`prepare(image, image_id)` → `segment(box_xyxy)`). `LocalSAMBackend` wraps a `SamPredictor` and caches by `image_id`. `RemoteSAMBackend` POSTs a pickled uint8 image + JSON coords to a gc-analysis-style SAM server; bounded by `DEFAULT_TIMEOUT_SECONDS=60` and treats the server URL as a trust boundary (response is pickle-loaded).

- **`_utils.py`** — `SAMSegmenter`: legacy wrapper around `SamPredictor` (used by `TraceAnything`); the new code path goes through `sam_backends.LocalSAMBackend`. Also contains `get_sam_model(name, device=None)` (process-wide weight cache shared by both widgets — predictors stay per-widget so `set_image` state doesn't collide), model download/caching (`~/.cache/napari-3d-Trace-Anything/`), image preprocessing, image type validation (`check_image_type`), and box generation from label regionprops (`create_boxes_list`).

- **`processing/clamp.py`** — CLAMP per-slice optimizer used by `TraceAnything._segment`: `optimize_slice` (multimask candidate choice by IoU with the previous mask, switch recovery, stray removal) and `inner_loop` (re-box and re-segment until IoU > 0.99 or 10 iterations; only with self-optimization on). Pure numpy over a `predict_fn`, so it is unit-tested without SAM. The outer loop (slice-to-slice propagation) is `TraceAnything._tracer` / `_predict`. Its output must stay identical to the paper's benchmark; do not change the algorithm or its constants.

### Bundled assets

- `model/vit/object_tracking_vittrack_2023sep.onnx` — VitTracker weights from OpenCV Zoo (Apache-2.0; <https://github.com/opencv/opencv_zoo>), bundled because `cv2.TrackerVit_Params.net` requires a real filesystem path. `importlib.resources.as_file` resolves it on editable / unpacked-wheel installs; zip-imported environments are not supported.

### Key Patterns

- **Slice propagation** (`TraceAnything`): Previous slice labels → bounding boxes → SAM segmentation on current slice → repeat
- **Frame tracking** (`TrackAnything`): VitTracker predicts next-frame box → SAM segments → optional frame-skip detection bridges occlusions
- **Threading**: Both widgets use `napari._qt.qthreading.create_worker()` for non-blocking batch processing
- **Device detection**: Auto-selects CUDA → MPS → CPU via torch
- **Instance tracking** (`TraceAnything` only): Each labeled object gets a unique integer label, maintained across slices

### Origins / parallel repos

`tracking.py`, `sam_backends.RemoteSAMBackend`, and `_track_widget.py` are adapted from `neurobiology-ut/napari-gc-analysis` (Apache-2.0). The upstream repo is still maintained in parallel; bug fixes that apply to both must be ported manually until a single canonical source is decided.

### Model checkpoint storage

SAM checkpoints download to `~/.cache/napari-3d-Trace-Anything/` atomically (`.part` → `os.replace`) and are validated with `zipfile.is_zipfile` on each load. If the cached file is corrupt (e.g. interrupted download from a previous run), `_utils.load_model` deletes it and re-downloads once before raising — the user doesn't need to clear the cache by hand. Concurrent loads of the same checkpoint URL (e.g. `default` and `vit_h` which share `sam_vit_h_4b8939.pth`) are serialized through a per-URL `RLock`, and the resulting SAM weights are cached by URL so two model names pointing at the same file share one in-memory instance.

### Known limitations

- **Frame-skip in `tracker` mode runs VitTracker twice per non-skipped frame** (`TrackAnything._advance_through_skips`): when skip detection is enabled and the method is `Tracker`, `check_frame_movement` predicts the new box once for the skip decision, and the outer `_trace` loop then predicts again on the main tracker to advance state. Reusing the temp tracker's bbox would discard the main tracker's incremental state; a cleaner fix requires splitting skip-detection and main-loop tracker state. ECC / POC / AKAZE skip methods are not affected.
- **macOS + MPS occasional segfault during long Trace**: reported but not reliably reproducible. PR#10 adds `qRegisterMetaType("QVector<int>")` at import as a defensive measure (the queued-connection warning that often precedes the crash) and verbose logging in `_segment_frame` so the next reproduction includes which frame the crash happened on. Investigation tracked separately; if you hit it, the terminal log right before the SIGSEGV is the most useful artifact.
- **Windows AV scanners can briefly hold downloaded checkpoints open**: `_utils._safe_replace` retries `os.replace` up to 3 times to absorb this. If it still fails, the `.pth` file may need to be excluded from the AV scanner.

## Code Style

- **Black**: line length 79, targets py38–py310
- **Ruff**: rules E, F, W, UP, I, BLE, B, A, C4, ISC, G, PIE, SIM (ignores E501, UP006, UP007, SIM117)
- Source lives under `src/napari_3d_trace_anything/`
