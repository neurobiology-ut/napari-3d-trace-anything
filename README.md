# napari-3d-trace-anything

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/napari-3d-trace-anything.svg?color=green)](https://github.com/neurobiology-ut/napari-3d-trace-anything/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-3d-trace-anything.svg?color=green)](https://pypi.org/project/napari-3d-trace-anything)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-3d-trace-anything.svg?color=green)](https://python.org)
[![tests](https://github.com/neurobiology-ut/napari-3d-trace-anything/workflows/tests/badge.svg)](https://github.com/neurobiology-ut/napari-3d-trace-anything/actions)
[![codecov](https://codecov.io/gh/neurobiology-ut/napari-3d-trace-anything/branch/main/graph/badge.svg)](https://codecov.io/gh/neurobiology-ut/napari-3d-trace-anything)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-3d-trace-anything)](https://napari-hub.org/plugins/napari-3d-trace-anything)

3D tracer using Segment Anything Model

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-3d-trace-anything` via [pip]:

    pip install napari-3d-trace-anything



To install latest development version :

    pip install git+https://github.com/neurobiology-ut/napari-3d-trace-anything.git


## Usage

### Setup

1. Open a 3D image stack in napari
2. Open the plugin from **Plugins > napari-3d-trace-anything**
3. Select a SAM model (e.g. `vit_h`) and click **load model** (the checkpoint is downloaded automatically on first use to `~/.cache/napari-3d-Trace-Anything/`)
4. Select your image in **input image layer** — the plugin will create **Predicted-Label** and **Merged-Label** layers automatically
5. Create or select an existing labels layer and set it as **output labels layer** — this layer is used as a reference for generating bounding boxes from the previous slice (trace results are written to **Predicted-Label**)
6. Set **merged labels layer** to the layer where you want to accumulate finalized results (default: **Merged-Label**)

### Basic tracing (instance mode OFF)

1. Set **start slice** and **end slice** for the range to trace
2. Switch to the **SAM-Box** shapes layer and draw a bounding box around the object on the start slice
3. Click **trace** — the plugin propagates the segmentation slice by slice using SAM, writing results to **Predicted-Label**
4. Press **A** to accept: the prediction is transferred to the merged labels layer with a new label number, and Predicted-Label is cleared
5. Repeat for other objects

### Instance mode (instance mode ON)

When **instance mode** is checked, each bounding box is assigned a specific label number that is maintained across slices.

1. Check the **instance mode** checkbox
2. Draw a bounding box on the SAM-Box layer — a dialog appears asking for the instance number
3. Draw additional boxes with different numbers if needed
4. Click **trace** — each object keeps its assigned number across slices
5. Press **A** to accept: label numbers are copied as-is to the merged labels layer (existing labels are not overwritten)

### Options

- **margin ratio**: Adjusts the bounding box margin for automatic box generation from previous slice labels. Positive values expand boxes, negative values shrink them (range: -1.0 to 1.0)
- **self-optimization**: When checked, runs CLAMP's inner loop on each slice (see [CLAMP](#clamp)). When unchecked, only the outer loop (slice-to-slice propagation) is applied
- **one box per label** (instance mode, on by default): Prompts each label with one box around all its pixels on the previous slice and writes only the largest component of the result, as in the paper. When unchecked, each separate blob gets its own box and all components are kept

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **A** | Accept prediction — transfer Predicted-Label to merged labels layer |
| **C** | Clear the current slice in the output labels layer |

### Workflow tips

- **Branching objects**: If a traced object splits into two, accept the first trace to merged, then re-trace from the branch point with the same label number. The `cleared_labels` mechanism in `_predict` will replace the old trace in Predicted-Label, while the accepted portion remains safe in the merged layer.
- **Correcting a trace**: Simply re-draw a box with the same instance number and trace again. The new result overwrites the old one in Predicted-Label. Accept when satisfied.
- **Large images**: Images larger than 1024x1024 are automatically cropped around each bounding box for SAM inference, so there is no need to manually resize.

## CLAMP

Tracing uses CLAMP (Closed-Loop Auto-Mask Propagation), which has two loops:

| Loop | What it does | Code |
|---|---|---|
| Outer | Propagates across slices: the previous slice's mask gives the prompt box for the next slice | `TraceAnything._tracer`, `_predict`, `create_boxes_list` |
| Inner | Refines one slice: re-box from the mask and re-segment until IoU > 0.99 or 10 iterations | `inner_loop`, called from `optimize_slice` |

In instance mode with default settings (margin ratio 0, all filters 0, one box per label), **self-optimization** on and off correspond to CLAMP and the outer-loop-only condition in the paper.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-3d-trace-anything" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/neurobiology-ut/napari-3d-trace-anything/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
