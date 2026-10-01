"""Tests for _utils."""

import numpy as np

from .._utils import create_boxes_list


def test_multiple_blobs_same_label():
    """同じラベル値を持つ複数のblobに対してバウンディングボックスが正しく生成されることを確認"""
    # テスト用のラベル画像を作成
    # 2つの離れたblobを含む30x30のラベル画像
    labels = np.zeros((30, 30), dtype=np.int32)
    # 1つ目のblob (ラベル1)
    labels[5:10, 5:10] = 1
    # 2つ目のblob (同じくラベル1)
    labels[20:25, 20:25] = 1

    # バウンディングボックスを生成
    boxes, label_values = create_boxes_list(labels)

    # 2つのblobに対して2つのバウンディングボックスが生成されることを確認
    assert (
        len(boxes) == 2
    ), "2つのblobに対して2つのバウンディングボックスが生成されるべき"

    # ラベル値が正しく対応していることを確認
    assert (
        len(label_values) == 2
    ), "2つのblobに対して2つのラベル値が存在するべき"
    assert all(
        val == 1 for val in label_values
    ), "すべてのラベル値が1であるべき"

    # それぞれのボックスが正しい形状を持つことを確認
    for box in boxes:
        assert box.shape == (
            4,
            3,
        ), "各ボックスは4つの点と3つの座標値(z,y,x)を持つべき"

    # Extents are inclusive pixel indices: the blob at [5:10, 5:10] spans
    # pixels 5..9, so its box runs 5..9. This is the convention the paper's
    # benchmark used for prompt boxes; an exclusive max (10) gives a box 1 px
    # larger, which changes SAM's output and the traced result.
    box1 = boxes[0]
    assert (
        box1[0][1] == 5 and box1[0][2] == 5
    ), "1つ目のblobの左上座標が正しくない"
    assert (
        box1[2][1] == 9 and box1[2][2] == 9
    ), "1つ目のblobの右下座標が正しくない"

    box2 = boxes[1]
    assert (
        box2[0][1] == 20 and box2[0][2] == 20
    ), "2つ目のblobの左上座標が正しくない"
    assert (
        box2[2][1] == 24 and box2[2][2] == 24
    ), "2つ目のblobの右下座標が正しくない"
