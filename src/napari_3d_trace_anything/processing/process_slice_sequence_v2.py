"""Process slice sequence with iterative optimization."""

import numpy as np
from skimage.measure import regionprops
from .._utils import SAMSegmenter, calculate_iou, create_box


def optimize_segmentation(
    image_slice,
    initial_box,
    segmenter,
    mergin_ratio=0.0,
):
    """1つのスライスに対してセグメンテーションを最適化する.

    boxとセグメンテーションを繰り返し最適化し、
    前回のセグメンテーション結果とのIoUが0.99を超えるまで処理を継続する.

    Args:
        image_slice (np.ndarray): 2D画像スライス
        initial_box (np.ndarray): 初期バウンディングボックス
        segmenter: SAMセグメンター
        mergin_ratio (float): バウンディングボックスのマージン比率

    Returns:
        tuple: (optimized_box, optimized_mask)
            - optimized_box: 最適化されたバウンディングボックス
            - optimized_mask: 最適化されたマスク
    """
    # 最初のセグメンテーション
    current_mask = segmenter.segment(image_slice, initial_box)
    current_box = initial_box
    
    iteration = 1
    prev_mask = None
    iou_score = 0
    
    # IoUが0.99を超えるまで繰り返し
    while True:
        print(f"  反復 {iteration}:")
        
        if prev_mask is not None:
            iou_score = calculate_iou(current_mask, prev_mask)
            print(f"    IoU: {iou_score:.4f}")
            
            if iou_score > 0.99:
                print("    収束条件を満たしました")
                break
        
        prev_mask = current_mask
        
        # 現在のマスクからRegionPropsを計算
        label_image = current_mask.astype(np.uint8)
        props = regionprops(label_image)[0]
        
        # 新しいboxを作成
        box_coords = create_box(props, mergin_ratio)
        current_box = box_coords
        
        # 新しいマスクを生成
        current_mask = segmenter.segment(image_slice, current_box)
        
        iteration += 1
        
        if iteration > 10:  # 最大反復回数
            print("    最大反復回数に達しました")
            break
    
    return current_box, current_mask


def process_slice_sequence_v2(
    image,
    initial_box,
    z_start,
    z_end,
    sam_predictor,
    mergin_ratio=0.0,
):
    """指定されたz範囲のスライスを処理する.

    各スライスでは、boxとセグメンテーションを繰り返し最適化する.
    前回のセグメンテーション結果とのIoUが0.99を超えるまで処理を継続する.

    Args:
        image (np.ndarray): 3D画像データ
        initial_box (np.ndarray): z_startスライスの初期box
        z_start (int): 開始スライス位置
        z_end (int): 終了スライス位置
        sam_predictor: SAMのpredictor
        mergin_ratio (float): バウンディングボックスのマージン比率

    Returns:
        tuple: (boxes_history, masks_history)
            - boxes_history: 各スライスの最終的なbox
            - masks_history: 各スライスの最終的なマスク

    Raises:
        ValueError: 無効なz範囲が指定された場合
    """
    # 入力値の検証
    if z_start < 0:
        raise ValueError("z_start must be non-negative")
    if z_end >= image.shape[0]:
        raise ValueError("z_end must be less than image depth")
    if z_end < z_start:
        raise ValueError("z_end must be greater than or equal to z_start")

    print(f"\n処理開始: z={z_start}から{z_end}のスライス")
    segmenter = SAMSegmenter(sam_predictor)
    
    boxes_history = []  # 各スライスの最終boxを保存
    masks_history = []  # 各スライスの最終マスクを保存
    current_box = initial_box  # 最初のboxを設定

    # 各スライスを処理
    for z in range(z_start, z_end + 1):
        print(f"\nスライス z={z} の処理:")
        
        # セグメンテーションを最適化
        current_box, current_mask = optimize_segmentation(
            image[z],
            current_box,
            segmenter,
            mergin_ratio
        )
        
        # 結果を保存
        boxes_history.append(current_box)
        masks_history.append(current_mask)
        
        # current_boxは既に最適化されているので、次のスライスの初期boxとしてそのまま使用
    
    return boxes_history, masks_history