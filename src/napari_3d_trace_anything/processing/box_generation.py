"""Box generation utilities for 3D tracing."""

import numpy as np
from tqdm import tqdm

from .._utils import calculate_iou


def generate_box_candidates(box):
    """Generate candidate boxes with various transformations.

    Args:
        box (np.ndarray): Input box with shape (4, 3) containing coordinates
            [[z, y1, x1], [z, y1, x2], [z, y2, x2], [z, y2, x1]]

    Returns:
        np.ndarray: Array of candidate boxes with shape (81, 4, 3)
    """
    # 元のボックスの中心と大きさを計算
    center_y = (box[0][1] + box[2][1]) / 2
    center_x = (box[0][2] + box[1][2]) / 2
    height = box[2][1] - box[0][1]
    width = box[1][2] - box[0][2]

    # スケール変換の係数
    scales = [0.9, 1.0, 1.1]

    # アスペクト比の変更係数（テストケースの順序に合わせる）
    aspects = [1.0, 1.1, 0.9]

    # 位置のオフセット（グリッドパターン、サイズの10%）
    y_offsets = [-0.1 * height, 0, 0.1 * height]
    x_offsets = [-0.1 * width, 0, 0.1 * width]

    # 候補ボックスを格納する配列の初期化
    n_scales = len(scales)
    n_aspects = len(aspects)
    n_offsets = len(y_offsets) * len(x_offsets)
    total_candidates = n_scales * n_aspects * n_offsets
    candidates = np.zeros((total_candidates, 4, 3))
    idx = 0

    # すべての組み合わせを生成
    # スケール -> アスペクト比 -> 位置の順で生成
    for scale_idx, scale in enumerate(scales):
        for aspect_idx, aspect in enumerate(aspects):
            for y_idx, y_off in enumerate(y_offsets):
                for x_idx, x_off in enumerate(x_offsets):
                    # インデックスの計算
                    y_offset_len = len(y_offsets)
                    x_offset_len = len(x_offsets)
                    idx = (
                        y_idx * x_offset_len + x_idx +
                        aspect_idx * y_offset_len * x_offset_len +
                        scale_idx * n_aspects * y_offset_len * x_offset_len
                    )
                    # 新しい中心位置
                    new_center_y = center_y + y_off
                    new_center_x = center_x + x_off

                    # デバッグ情報
                    print(f"\nBox {idx}:")
                    print(f"Scale: {scale}, Aspect: {aspect}")

                    # 新しい大きさの計算
                    new_height = height * scale
                    new_width = width * scale

                    # アスペクト比による調整
                    new_width = new_width * aspect

                    print(f"Original size: {height}x{width}")
                    print(f"New size: {new_height}x{new_width}")
                    print(f"Actual aspect ratio: {new_width/new_height}")

                    # 新しいボックスの座標を計算
                    half_height = new_height / 2
                    half_width = new_width / 2
                    z = box[0][0]

                    # ボックスの各頂点を計算
                    y_min = new_center_y - half_height
                    y_max = new_center_y + half_height
                    x_min = new_center_x - half_width
                    x_max = new_center_x + half_width

                    new_box = np.array([
                        [z, y_min, x_min],
                        [z, y_min, x_max],
                        [z, y_max, x_max],
                        [z, y_max, x_min]
                    ])

                    candidates[idx] = new_box
                    idx += 1

    return candidates


def select_top_boxes(prev_masks, current_masks, boxes, top_k=3):
    """前後スライス間のIoUに基づいて上位のboxを選択.

    Args:
        prev_masks: 前スライスのマスク
        current_masks: 現在のマスク
        boxes: ボックス座標
        top_k: 選択数

    Returns:
        tuple: (boxes, masks) 選択結果
    """
    if len(prev_masks) != 1:
        raise ValueError("Previous masks should contain exactly one mask")

    prev_mask = prev_masks[0]

    # 各ボックスのIoUスコアを計算
    print("\nIoUスコアの計算:")
    iou_scores = []
    for i, curr_mask in enumerate(current_masks):
        iou = calculate_iou(prev_mask, curr_mask)
        iou_scores.append(iou)
        print(f"  マスク {i+1}: IoU = {iou:.3f}")

    # スコアの降順でソート
    indices = np.argsort(iou_scores)[::-1]

    # 上位k個を選択
    selected_indices = indices[:top_k]
    selected_boxes = boxes[selected_indices]
    selected_masks = [current_masks[i] for i in selected_indices]

    return selected_boxes, selected_masks


def process_slice_sequence(
        image,
        initial_box,
        z_start,
        sam_predictor,
        top_k=3):
    """3Dトレース用のスライス処理.

    Args:
        image: 3D画像
        initial_box: 初期box
        z_start: 開始Z位置
        sam_predictor: SAM
        top_k: 選択数

    Returns:
        tuple: box/maskの履歴
    """
    print(f"\n処理開始: z={z_start}のスライスから")
    from .._utils import SAMSegmenter
    segmenter = SAMSegmenter(sam_predictor)

    # 最初のスライスの処理
    initial_mask = segmenter.segment(
        image[z_start],
        initial_box
    )
    current_boxes = [initial_box]
    current_masks = [initial_mask]

    # 次のスライスの処理
    z = z_start + 1
    selected_boxes_history = []
    selected_masks_history = []

    while z < min(z_start + 3, image.shape[0]):  # 最大2スライス先まで
        print(f"\nスライス z={z} の処理:")
        next_boxes = []
        next_masks = []

        # 前のスライスの各ボックスに対して候補を生成
        for box_idx, prev_box in enumerate(current_boxes):
            print(f"\nボックス {box_idx + 1}/{len(current_boxes)} の候補を生成中...")
            candidates = generate_box_candidates(prev_box)

            # 各候補に対してセグメンテーション
            print("各候補に対してセグメンテーションを実行中...")
            for box in tqdm(candidates, desc=f"Box {box_idx + 1} Candidates"):
                # SAMが期待する形式 [x1, y1, x2, y2] に変換
                sam_box = np.array([
                    box[0][2],  # x1
                    box[0][1],  # y1
                    box[2][2],  # x2
                    box[2][1]   # y2
                ])
                mask = segmenter.segment(
                    image[z],
                    sam_box)
                next_boxes.append(box)
                next_masks.append(mask)

        # 前のスライスの各マスクに対して、次のスライスの候補と比較
        all_selected_boxes = []
        all_selected_masks = []
        for prev_mask in current_masks:
            selected_boxes, selected_masks = select_top_boxes(
                [prev_mask],
                next_masks,
                np.array(next_boxes),
                top_k
            )
            all_selected_boxes.extend(selected_boxes)
            all_selected_masks.extend(selected_masks)

        # 次のスライスの準備
        selected_boxes_history.append(all_selected_boxes)
        selected_masks_history.append(all_selected_masks)
        current_boxes = all_selected_boxes
        current_masks = all_selected_masks
        z += 1

    return selected_boxes_history, selected_masks_history
