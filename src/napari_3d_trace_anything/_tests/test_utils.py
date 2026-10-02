"""Tests for _utils."""

import numpy as np

from .._utils import create_boxes_list


def test_multiple_blobs_same_label():
    """Each blob of a label gets its own box with the right corners."""
    # 30x30 label image with two separate blobs of label 1
    labels = np.zeros((30, 30), dtype=np.int32)
    # first blob
    labels[5:10, 5:10] = 1
    # second blob
    labels[20:25, 20:25] = 1

    boxes, label_values = create_boxes_list(labels)

    assert (
        len(boxes) == 2
    ), "expected one box per blob"

    assert (
        len(label_values) == 2
    ), "expected one label value per blob"
    assert all(
        val == 1 for val in label_values
    ), "all label values should be 1"

    for box in boxes:
        assert box.shape == (
            4,
            3,
        ), "each box has four (z, y, x) vertices"

    # Extents are inclusive pixel indices: the blob at [5:10, 5:10] spans
    # pixels 5..9, so its box runs 5..9. This is the convention the paper's
    # benchmark used for prompt boxes; an exclusive max (10) gives a box 1 px
    # larger, which changes SAM's output and the traced result.
    box1 = boxes[0]
    assert (
        box1[0][1] == 5 and box1[0][2] == 5
    ), "wrong top-left corner of the first box"
    assert (
        box1[2][1] == 9 and box1[2][2] == 9
    ), "wrong bottom-right corner of the first box"

    box2 = boxes[1]
    assert (
        box2[0][1] == 20 and box2[0][2] == 20
    ), "wrong top-left corner of the second box"
    assert (
        box2[2][1] == 24 and box2[2][2] == 24
    ), "wrong bottom-right corner of the second box"
