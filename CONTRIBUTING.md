# Contributing

## Setup

```bash
pip install -e ".[testing]"
```

## Tests and lint

```bash
pytest -v --cov=napari_3d_trace_anything --cov-report=xml
pytest src/napari_3d_trace_anything/_tests/test_clamp.py -v   # one file
tox                                                           # py3.10–3.13

black src/napari_3d_trace_anything
ruff check src/napari_3d_trace_anything --fix
pre-commit run --all-files
```

- Black and Ruff use line length 79 and target Python 3.10+.
- `ruff` is configured with `fix = true`, so a plain `ruff check` rewrites files. Use `--no-fix` to only report.
- CI runs the test suite on Linux, macOS and Windows for pull requests to `dev` and `main`.
- Work goes to `dev` through pull requests and is released to `main` from `dev`.

## Layout

The plugin registers two widgets through `napari.yaml`. They are independent and use disjoint layer names, so both can run side by side.

| Widget | Use case | Box layer | Output layer(s) |
|---|---|---|---|
| `_widget.py:TraceAnything` | 3D z-stack: propagate labels slice to slice | `SAM-Box` | `Predicted-Label`, `Labels-{img}`, `Merged-Label-{img}` |
| `_track_widget.py:TrackAnything` | Time series: track and segment one object across frames | `Track-Box-{img}` | `Track-Label-{img}` |

Modules:

- **`_widget.py`** — `TraceAnything`. `_tracer` steps through the slices (CLAMP's outer loop) in a worker thread; `_predict` builds the prompt boxes from the previous slice, crops a 1024×1024 ROI on large images, and calls `_segment`, which runs `clamp.optimize_slice`. In instance mode with **One Box per Label** on, each label propagates from its own full mask and only its largest component is written to the layer.
- **`processing/clamp.py`** — the CLAMP per-slice optimizer: `optimize_slice` (candidate choice by IoU with the previous mask, switch recovery, stray removal) and `inner_loop` (re-box and re-segment until IoU > 0.99 or 10 iterations; only with Self-Optimization on). It is pure numpy over a `predict_fn`, so it is unit-tested without SAM.
- **`_track_widget.py`** — `TrackAnything`. OpenCV `TrackerVit` moves the box from frame to frame and SAM segments each box. Local or remote SAM, optional frame-skip detection (ECC / POC / AKAZE / Tracker) for occlusions, and a Segment Only mode for existing boxes.
- **`tracking.py`** — pure functions for `TrackAnything`: the VitTracker loader, displacement estimators, ROI helpers and the frame-skip dispatcher.
- **`sam_backends.py`** — `SAMBackend` protocol (`prepare(image, image_id)` → `segment(box_xyxy)`) with `LocalSAMBackend` and `RemoteSAMBackend`. The remote backend pickle-loads the server's response, so the server URL is a trust boundary.
- **`_utils.py`** — `SAMSegmenter` (wraps `SamPredictor`; `make_predict_fn` feeds `clamp.optimize_slice`), the shared model cache `get_sam_model`, checkpoint download, image preprocessing, and `create_boxes_list` (prompt boxes from label images).

## CLAMP must match the paper

`processing/clamp.py`, its constants, and the box conventions in `_predict` and `create_boxes_list` reproduce the published benchmark pixel for pixel. Do not change the algorithm, its constants, or these conventions. In particular, `create_boxes_list` uses inclusive pixel extents for the next slice's box, while `create_box` (used inside the inner loop) stays exclusive; a 1 px difference changes SAM's output and the trace.

## Model checkpoints

SAM checkpoints download to `~/.cache/napari-3d-Trace-Anything/` atomically (`.part`, then `os.replace`) and are checked with `zipfile.is_zipfile` on every load. A corrupt cached file is deleted and downloaded again once. Loads of the same checkpoint URL are serialized with a per-URL lock, and model names that share a file share one in-memory model. Predictors stay per widget so their `set_image` state does not collide.

`model/vit/object_tracking_vittrack_2023sep.onnx` is the VitTracker model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0). It is bundled because `cv2.TrackerVit_Params.net` needs a real file path; zip-imported installs are not supported.

## Known limitations

- **Frame skip in Tracker mode runs VitTracker twice per frame.** The skip check predicts once and the main loop predicts again. Reusing the first prediction would discard the main tracker's state; fixing it needs separate tracker state for skip detection.
- **Occasional segfault on macOS with MPS during long traces.** Not reliably reproducible. `QVector<int>` is registered at import as a precaution, and `_segment_frame` logs each frame so the terminal output before a crash shows where it happened.
- **Windows antivirus scanners can hold a fresh checkpoint open.** `_utils._safe_replace` retries `os.replace` up to three times; if it still fails, exclude the `.pth` file from scanning.
- **On Windows, torch must be imported before Qt.** Otherwise `c10.dll` fails to initialize (WinError 1114). The test suite does this in `conftest.py`; whether the napari GUI is affected has not been checked.
- **Resuming a trace in One Box per Label mode** starts from the largest component written to the layer, so it can differ from a single uninterrupted trace.
