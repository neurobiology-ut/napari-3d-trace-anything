"""Test for box generation functionality."""

import os

import numpy as np
import tifffile
import torch
from segment_anything import (
    SamPredictor,
)
from segment_anything import (
    sam_model_registry as sam_registry,
)

from .._utils import create_boxes_list
from ..processing.box_generation import (
    generate_box_candidates,
    process_slice_sequence,
)


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
    assert len(boxes) == 2, "2つのblobに対して2つのバウンディングボックスが生成されるべき"

    # ラベル値が正しく対応していることを確認
    assert len(label_values) == 2, "2つのblobに対して2つのラベル値が存在するべき"
    assert all(val == 1 for val in label_values), "すべてのラベル値が1であるべき"

    # それぞれのボックスが正しい形状を持つことを確認
    for box in boxes:
        assert box.shape == (4, 3), "各ボックスは4つの点と3つの座標値(z,y,x)を持つべき"

    # 1つ目のblobのバウンディングボックス
    box1 = boxes[0]
    assert box1[0][1] == 5 and box1[0][2] == 5, "1つ目のblobの左上座標が正しくない"
    assert box1[2][1] == 10 and box1[2][2] == 10, "1つ目のblobの右下座標が正しくない"

    # 2つ目のblobのバウンディングボックス
    box2 = boxes[1]
    assert box2[0][1] == 20 and box2[0][2] == 20, "2つ目のblobの左上座標が正しくない"
    assert box2[2][1] == 25 and box2[2][2] == 25, "2つ目のblobの右下座標が正しくない"


def test_box_candidates_count():
    """生成される候補ボックスの数が正しいことを確認"""
    # テスト用の初期ボックス
    test_box = np.array([
        [0, 10, 10],  # z, y1, x1
        [0, 10, 20],  # z, y1, x2
        [0, 20, 20],  # z, y2, x2
        [0, 20, 10],  # z, y2, x1
    ])

    candidates = generate_box_candidates(test_box)

    # 3(スケール) x 3(アスペクト比) x 9(位置) = 81個の候補
    assert candidates.shape == (81, 4, 3)


def test_box_transformations():
    """各変換が正しく適用されていることを確認"""
    # テスト用の初期ボックス（10x10の正方形）
    test_box = np.array([
        [0, 10, 10],
        [0, 10, 20],
        [0, 20, 20],
        [0, 20, 10],
    ])

    candidates = generate_box_candidates(test_box)

    # 元のボックスの中心
    original_center_y = 15
    original_center_x = 15

    # スケール0.9のボックスをチェック
    scaled_box = candidates[0]  # 最初の候補（0.9倍、アスペクト比1.0）
    height = scaled_box[2][1] - scaled_box[0][1]
    width = scaled_box[1][2] - scaled_box[0][2]

    assert np.isclose(height, 9.0)  # 10 * 0.9
    assert np.isclose(width, 9.0)   # 10 * 0.9

    # アスペクト比1.1のボックスをチェック
    # （スケール0.9、アスペクト比1.1のグループの最初）
    aspect_box = candidates[9]
    height = aspect_box[2][1] - aspect_box[0][1]
    width = aspect_box[1][2] - aspect_box[0][2]

    assert np.isclose(width / height, 1.1)  # アスペクト比

    # 位置オフセットのチェック
    # （スケール0.9、アスペクト比1.0の3番目のボックス）
    offset_box = candidates[2]
    center_y = (offset_box[0][1] + offset_box[2][1]) / 2
    center_x = (offset_box[0][2] + offset_box[1][2]) / 2

    assert not np.isclose(center_y, original_center_y)  # 中心位置が変化
    assert not np.isclose(center_x, original_center_x)  # 中心位置が変化


def test_box_coordinates():
    """生成されたボックスの座標が正しい形式であることを確認"""
    test_box = np.array([
        [0, 10, 10],
        [0, 10, 20],
        [0, 20, 20],
        [0, 20, 10],
    ])

    candidates = generate_box_candidates(test_box)

    for box in candidates:
        # zの値が保持されていることを確認
        assert np.all(box[:, 0] == 0)

        # 座標の接続が正しいことを確認
        assert np.isclose(box[0][1], box[1][1])  # y1は同じ
        assert np.isclose(box[2][1], box[3][1])  # y2は同じ
        assert np.isclose(box[1][2], box[2][2])  # x2は同じ
        assert np.isclose(box[0][2], box[3][2])  # x1は同じ


def test_process_slice_sequence():
    """スライスシーケンス処理のテスト"""
    # SAMモデルのセットアップ
    model_type = "vit_h"
    sam_checkpoint = os.path.expanduser(
        "~/.cache/napari-3d-Trace-Anything/sam_vit_h_4b8939.pth"
    )
    sam = sam_registry[model_type](checkpoint=sam_checkpoint)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # テスト画像の読み込み
    here = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    image_path = os.path.join(root_dir, "test-data", "cropped_em.tif")
    image = tifffile.imread(image_path)

    # 初期ボックスの設定
    initial_box = np.array([
        [34, 370, 770],  # z, y1, x1
        [34, 370, 825],  # z, y1, x2
        [34, 450, 825],  # z, y2, x2
        [34, 450, 770],  # z, y2, x1
    ])

    # スライス処理の実行
    boxes_history, masks_history = process_slice_sequence(
        image, initial_box, 34, predictor
    )

    # 結果の検証
    assert len(boxes_history) == 2  # 2スライス分の履歴
    assert len(masks_history) == 2  # 2スライス分の履歴

    # 最初のスライス（z+1）では3つのボックスが選択される
    assert len(boxes_history[0]) == 3
    assert len(masks_history[0]) == 3

    # 次のスライス（z+2）では前のスライスの各ボックスに対して3つずつ選択される（計9個）
    assert len(boxes_history[1]) == 9
    assert len(masks_history[1]) == 9

    # すべてのボックスの形状を確認
    for boxes in boxes_history:
        for box in boxes:
            assert box.shape == (4, 3)

    # ボックス位置を描画した画像を保存
    initial_z = 34  # テストで使用するZ位置
    for z_idx, boxes in enumerate(boxes_history):
        current_z = initial_z + z_idx + 1  # 次のスライスのZ位置
        img = image[current_z].copy()  # 次のスライスの画像
        # RGB画像に変換
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)

        # すべての候補ボックスを描画（青色）
        if z_idx == 0:  # 最初のスライスの場合
            candidates = generate_box_candidates(initial_box)
        else:  # 次のスライスの場合
            candidates = []
            for prev_box in boxes_history[z_idx - 1]:
                candidates.extend(generate_box_candidates(prev_box))
            candidates = np.array(candidates)

        # 候補ボックスを青色で描画
        for box in candidates:
            y1, x1 = int(box[0][1]), int(box[0][2])
            y2, x2 = int(box[2][1]), int(box[2][2])

            # 青色でボックスを描画
            img[y1:y2, x1-1:x1+1] = [0, 0, 255]  # 左辺
            img[y1:y2, x2-1:x2+1] = [0, 0, 255]  # 右辺
            img[y1-1:y1+1, x1:x2] = [0, 0, 255]  # 上辺
            img[y2-1:y2+1, x1:x2] = [0, 0, 255]  # 下辺

        # 選択されたボックスを赤色で描画（上書き）
        for box in boxes:
            y1, x1 = int(box[0][1]), int(box[0][2])
            y2, x2 = int(box[2][1]), int(box[2][2])

            # 赤色でボックスを描画
            img[y1:y2, x1-1:x1+1] = [255, 0, 0]  # 左辺
            img[y1:y2, x2-1:x2+1] = [255, 0, 0]  # 右辺
            img[y1-1:y1+1, x1:x2] = [255, 0, 0]  # 上辺
            img[y2-1:y2+1, x1:x2] = [255, 0, 0]  # 下辺

        # 画像を保存（ファイル名にスライス番号を含める）
        output_path = os.path.join(
            root_dir,
            "test-data",
            f"slice_{current_z}_boxes.tif"
        )
        tifffile.imwrite(output_path, img)
