"""Test cases for process_slice_sequence_v2."""

import os
import numpy as np
import pytest
import torch
import tifffile
from segment_anything import (
    sam_model_registry as sam_registry,
    SamPredictor,
)

from ..processing.process_slice_sequence_v2 import process_slice_sequence_v2


def render_slice_results(image_slice, box, mask):
    """処理結果の可視化を行う"""
    rendered = image_slice.copy()
    if len(rendered.shape) == 2:
        rendered = np.stack([rendered] * 3, axis=-1)
    
    # マスクをオーバーレイ（半透明の赤）
    mask_overlay = np.zeros_like(rendered)
    mask_overlay[mask] = [255, 0, 0]  # 赤色
    rendered = rendered * 0.7 + mask_overlay * 0.3
    
    # boxを描画（青色）
    x1, y1, x2, y2 = box
    rendered = rendered.astype(np.uint8)
    rendered[int(y1):int(y1)+2, int(x1):int(x2), :] = [0, 0, 255]  # 上辺
    rendered[int(y2)-2:int(y2), int(x1):int(x2), :] = [0, 0, 255]  # 下辺
    rendered[int(y1):int(y2), int(x1):int(x1)+2, :] = [0, 0, 255]  # 左辺
    rendered[int(y1):int(y2), int(x2)-2:int(x2), :] = [0, 0, 255]  # 右辺
    
    return rendered


def create_mock_predictor():
    """モック化されたSAM predictorを作成"""
    class MockPredictor:
        def set_image(self, image):
            pass
            
        def predict(self, box, multimask_output=False):
            # 単純な円形のマスクを生成
            h, w = 100, 100
            Y, X = np.ogrid[:h, :w]
            center = (box[0][1] + box[0][3]) // 2, (box[0][0] + box[0][2]) // 2
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = dist_from_center <= 20
            return np.array([mask]), None, None
            
    return MockPredictor()


def test_process_slice_sequence_v2_with_mock():
    """モックデータを使用した基本的な機能テスト"""
    # テスト用の3D画像を作成
    image = np.zeros((10, 100, 100))
    
    # 初期boxを設定
    initial_box = np.array([40, 40, 60, 60])  # [y1, x1, y2, x2]
    
    # モックpredictor
    mock_predictor = create_mock_predictor()
    
    # プロセスを実行
    boxes_history, masks_history = process_slice_sequence_v2(
        image=image,
        initial_box=initial_box,
        z_start=34,
        z_end=38,
        sam_predictor=mock_predictor
    )
    
    # 結果の検証
    assert len(boxes_history) == 5  # z=34から38までの5スライス
    assert len(masks_history) == 5
    
    # 各マスクが2D配列であることを確認
    for mask in masks_history:
        assert len(mask.shape) == 2
        assert mask.dtype == bool
    
    # 各boxが正しい形式であることを確認
    for box in boxes_history:
        assert len(box) == 4  # [y1, x1, y2, x2]
        assert box[2] > box[0]  # y2 > y1
        assert box[3] > box[1]  # x2 > x1


def test_process_slice_sequence_v2_with_real_data():
    """実際の画像とSAMモデルを使用したテスト"""
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
    image_path = os.path.join(root_dir, "test-data3", "cropped_em.tif")
    image = tifffile.imread(image_path)

    # 初期ボックスの設定
    # initial_box = np.array([770, 370, 825, 450])  # [x1, y1, x2, y2]
    initial_box = np.array([1370, 1120, 1425, 1190])  # [x1, y1, x2, y2]

    start_z = 0
    end_z = 34

    # 画像サイズの確認とクロップ処理
    height, width = image.shape[1:]
    should_crop = width > 1024 or height > 1024

    if should_crop:
        # initial_boxの中心座標を計算 (x1, y1, x2, y2の順)
        center_x = (initial_box[0] + initial_box[2]) // 2  # x座標の中心
        center_y = (initial_box[1] + initial_box[3]) // 2  # y座標の中心

        print(f"Center coordinates: ({center_x}, {center_y})")
        print(f"Original image size: {width}x{height}")

        # クロップ範囲を計算（1024x1024を確保)
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

        # クロップされた画像を作成
        cropped_image = image[:, y1:y2, x1:x2]

        # initial_boxの座標をクロップ後の座標系に変換 (x1, y1, x2, y2の順)
        cropped_initial_box = np.array([
            initial_box[0] - x1,  # x1
            initial_box[1] - y1,  # y1
            initial_box[2] - x1,  # x2
            initial_box[3] - y1   # y2
        ])

        print(f"Crop region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Original box: {initial_box}")
        print(f"Cropped box: {cropped_initial_box}")
        print(f"Cropped image size: {cropped_image.shape}")

        # クロップされた画像でスライス処理を実行
        boxes_history, masks_history = process_slice_sequence_v2(
            image=cropped_image,
            initial_box=cropped_initial_box,
            z_start=start_z,
            z_end=end_z,
            sam_predictor=predictor,
            mergin_ratio=0.15
        )

        # boxの座標をオリジナルの座標系に戻す
        boxes_history = [
            np.array([box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1])
            for box in boxes_history
        ]

        # マスクをオリジナルサイズの画像に戻す
        full_size_masks = []
        for mask in masks_history:
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask
            full_size_masks.append(full_mask)
        masks_history = full_size_masks
    else:
        # クロップが不要な場合は通常通り処理
        boxes_history, masks_history = process_slice_sequence_v2(
            image=image,
            initial_box=initial_box,
            z_start=start_z,
            z_end=end_z,
            sam_predictor=predictor,
            mergin_ratio=0.15
        )

    # 結果の検証
    assert len(boxes_history) == end_z - start_z + 1
    assert len(masks_history) == end_z - start_z + 1

    # レンダリング結果を保存
    for z_idx, (box, mask) in enumerate(zip(boxes_history, masks_history)):
        current_z = start_z + z_idx
        img = image[current_z]
        rendered = render_slice_results(img, box, mask)
        
        # 画像を保存
        output_path = os.path.join(
            root_dir,
            "test-data3",
            f"slice_{current_z}_boxes_v2.tif"
        )
        tifffile.imwrite(output_path, rendered)


def test_convergence():
    """収束テスト"""
    # テスト用の3D画像を作成
    image = np.zeros((5, 100, 100))
    
    class ConvergencePredictor:
        def __init__(self):
            self.call_count = 0
            
        def set_image(self, image):
            pass
            
        def predict(self, box, multimask_output=False):
            self.call_count += 1
            # 毎回同じマスクを返して即時収束させる
            mask = np.zeros((100, 100), dtype=bool)
            mask[40:60, 40:60] = True
            return np.array([mask]), None, None
    
    predictor = ConvergencePredictor()
    initial_box = np.array([40, 40, 60, 60])
    
    # プロセスを実行
    boxes_history, masks_history = process_slice_sequence_v2(
        image=image,
        initial_box=initial_box,
        z_start=34,
        z_end=34,  # 1スライスのみテスト
        sam_predictor=predictor
    )
    
    # 2回目の反復で収束するはず（同じマスクを返すため）
    assert predictor.call_count == 2


def test_invalid_z_range():
    """無効なz範囲のテスト"""
    image = np.zeros((10, 100, 100))
    initial_box = np.array([40, 40, 60, 60])
    mock_predictor = create_mock_predictor()
    
    # z_end < z_start
    with pytest.raises(ValueError):
        process_slice_sequence_v2(
            image=image,
            initial_box=initial_box,
            z_start=5,
            z_end=3,
            sam_predictor=mock_predictor
        )
    
    # z_start < 0
    with pytest.raises(ValueError):
        process_slice_sequence_v2(
            image=image,
            initial_box=initial_box,
            z_start=-1,
            z_end=5,
            sam_predictor=mock_predictor
        )
    
    # z_end >= image.shape[0]
    with pytest.raises(ValueError):
        process_slice_sequence_v2(
            image=image,
            initial_box=initial_box,
            z_start=0,
            z_end=10,
            sam_predictor=mock_predictor
        )