import os
import urllib

import numpy as np
from skimage.color import gray2rgb
from skimage.measure import label, regionprops


class SAMSegmenter:
    """SAMを使用したセグメンテーションを行うクラス"""

    def __init__(self, predictor):
        """
        Args:
            predictor (SamPredictor): SAMのpredictor
        """
        if not (hasattr(predictor, "set_image") and hasattr(predictor, "predict")):
            raise ValueError("predictor must have set_image and predict methods")
        self.predictor = predictor
        self.current_image = None

    def segment(self, image, box):
        """画像のセグメンテーションを行う

        Args:
            image (np.ndarray): 入力画像
            box (np.ndarray or list): バウンディングボックス

        Returns:
            np.ndarray: セグメンテーションマスク
        """
        # boxをnumpy配列に変換
        box = np.array(box)

        # グレースケール画像の場合、RGB形式に変換
        if len(image.shape) == 2:
            image = gray2rgb(image)

        # 画像が前回と異なる場合のみset_imageを実行
        if self.current_image is None or not np.array_equal(
            image, self.current_image
        ):
            self.predictor.set_image(image)
            self.current_image = image

        # ボックスプロンプトの変換
        # box形式1: [[z, y1, x1], [z, y1, x2], [z, y2, x2], [z, y2, x1]]
        # box形式2: [x1, y1, x2, y2]
        if len(box.shape) == 2:  # 形式1の場合
            input_box = np.array(
                [
                    box[0, 2],  # x1
                    box[0, 1],  # y1
                    box[2, 2],  # x2
                    box[2, 1],  # y2
                ]
            )
        else:  # 形式2の場合
            input_box = box

        # マスクの生成
        masks, _, _ = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )

        return masks[0]


def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks.

    Args:
        mask1 (np.ndarray): First binary mask
        mask2 (np.ndarray): Second binary mask

    Returns:
        float: IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def create_box(props, mergin_ratio=0.0):
    """RegionPropertiesオブジェクトからバウンディングボックスを作成

    Args:
        props: skimage.measure.RegionProperties object
        mergin_ratio (float): バウンディングボックスのマージン比率

    Returns:
        list: [x1, y1, x2, y2]形式のバウンディングボックス
    """
    minr, minc, maxr, maxc = props.bbox
    mergin_r = (maxr - minr) * mergin_ratio
    mergin_c = (maxc - minc) * mergin_ratio
    return [minc - mergin_c, minr - mergin_r, maxc + mergin_c, maxr + mergin_r]
    # return [minc, minr, maxc, maxr]  # x1, y1, x2, y2の順序


def create_boxes_list(
    labels, margin_ratio=0.0, max_objects=0, min_area=0
):
    """ラベル画像から複数のバウンディングボックスを作成

    Args:
        labels: ラベル付けされた画像。同じラベル値を持つ複数のblobが存在する場合、
               それぞれのblobに対して個別のバウンディングボックスが生成されます。
        margin_ratio (float): バウンディングボックスのマージン比率
        max_objects (int): ラベルごとの最大オブジェクト数 (0=無制限)
        min_area (int): 最小面積閾値 (0=フィルタなし)

    Returns:
        tuple: (boxes, label_values)
            boxes: [[z, y1, x1], [z, y1, x2], [z, y2, x2], [z, y2, x1]]
                形式のバウンディングボックスのリスト。各要素はnp.array
            label_values: 各バウンディングボックスに対応する元のラベル値のリスト
    """
    boxes = []
    label_values = []

    # ユニークなラベル値を取得（0は背景として除外）
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # 各ラベル値について処理
    for label_val in unique_labels:
        # 現在のラベル値のマスクを作成
        binary_mask = (labels == label_val)
        # 各blobを個別にラベリング
        components = label(binary_mask)
        props_list = list(regionprops(components))

        # 面積閾値フィルタ
        if min_area > 0:
            props_list = [
                p for p in props_list if p.area >= min_area
            ]

        # 面積降順ソート → Top-N (sort only when needed)
        if max_objects > 0:
            props_list = sorted(
                props_list, key=lambda p: p.area, reverse=True
            )
            props_list = props_list[:max_objects]

        # 各blobに対してバウンディングボックスを生成
        for props in props_list:
            box_coords = create_box(props, margin_ratio)
            box = np.array([
                [0, box_coords[1], box_coords[0]],  # [z, y1, x1]
                [0, box_coords[1], box_coords[2]],  # [z, y1, x2]
                [0, box_coords[3], box_coords[2]],  # [z, y2, x2]
                [0, box_coords[3], box_coords[0]],  # [z, y2, x1]
            ])
            boxes.append(box)
            label_values.append(label_val)

    return boxes, label_values


def segment_with_sam(predictor, image, box):
    """SAMを使用して画像のセグメンテーションを行う

    Args:
        predictor (SamPredictor): SAMのpredictor
        image (np.ndarray): 入力画像
        box (list): [x1, y1, x2, y2]形式のバウンディングボックス

    Returns:
        np.ndarray: セグメンテーションマスク
    """
    # グレースケール画像の場合、RGB形式に変換
    if len(image.shape) == 2:
        image = gray2rgb(image)

    # 画像をセット
    predictor.set_image(image)

    # マスクの生成
    masks, _, _ = predictor.predict(
        box=np.array(box)[None, :], multimask_output=False
    )

    return masks[0]


def load_model(model_name):
    """Load model

    Args:
        model_name (str): model name

    :return: model
    """
    model_urls = dict(
        default=(
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"
        ),
        vit_h=(
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"
        ),
        vit_l=(
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_l_0b3195.pth"
        ),
        vit_b=(
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_b_01ec64.pth"
        ),
    )
    model_url = model_urls[model_name]
    model_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "napari-3d-Trace-Anything",
        os.path.basename(model_url),
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        autodownload(model_url)
    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_name](checkpoint=model_path)
    return sam


def autodownload(model_url):
    """Download model

    Args:
        model_url (str): model url

    """

    urllib.request.urlretrieve(
        model_url,
        os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "napari-3d-Trace-Anything",
            os.path.basename(model_url),
        ),
    )


def preprocess(image, image_type, slice_index):
    """画像の前処理を行う

    Args:
        image (np.ndarray or dask.array): 入力画像
        image_type (str): 画像タイプ
        slice_index (int): スライスインデックス

    Returns:
        np.ndarray: 前処理された画像
    """
    if "stack" in image_type:
        slice_data = image[slice_index]
        if hasattr(slice_data, "compute"):
            return slice_data.compute()
        return slice_data
    if hasattr(image, "compute"):
        return image.compute()
    return image


def check_image_type(viewer, layer_name):
    image = viewer.layers[layer_name].data
    print(f"current image shape = {image.shape}")
    if len(image.shape) == 2 or len(image.shape) > 4 or (len(image.shape) == 3) & (image.shape[-1] == 4) or (len(image.shape) == 3) & (image.shape[-1] == 1) or (len(image.shape) == 3) & (image.shape[-1] == 2):  # Gray
        return "Not supported"
    elif (len(image.shape) == 3) & (
        image.shape[-1] > 4
    ):  # maybe stacked gray images
        return "stacked gray images"
    elif (len(image.shape) == 4) & (
        image.shape[-1] == 1
    ):  # maybe stacked gray images
        return "stacked gray images with channel"
    elif (len(image.shape) == 4) & (
        image.shape[-1] == 3
    ):  # maybe stacked RGB images
        return "stacked RGB images"
    elif (len(image.shape) == 4) & (image.shape[-1] == 2) or (len(image.shape) == 4) & (image.shape[-1] > 4) or (len(image.shape) == 3) & (image.shape[-1] == 3):
        return "Not supported"
    else:
        return "Not supported"


def parse_slice_range(range_str):
    """Parse a range string into a sorted list of unique slice numbers.

    Examples:
        "1-3, 5, 9-10" -> [1, 2, 3, 5, 9, 10]
        "" -> []

    Args:
        range_str (str): Range string (e.g., "1-3, 5, 9-10")

    Returns:
        list: Sorted list of unique slice numbers

    Raises:
        ValueError: If the range string is invalid
    """
    if not range_str or not range_str.strip():
        return []

    slice_numbers = []
    parts = [part.strip() for part in range_str.split(",")]

    for part in parts:
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start > end:
                    raise ValueError(
                        f"Invalid range: {part} (start > end)"
                    )
                if (end - start) > 100_000:
                    raise ValueError(
                        f"Range too large: {part}"
                    )
                slice_numbers.extend(range(start, end + 1))
            except ValueError as e:
                if "Invalid range" in str(e) or (
                    "Range too large" in str(e)
                ):
                    raise
                raise ValueError(
                    f"Invalid range: {part}"
                ) from e
        else:
            try:
                slice_numbers.append(int(part.strip()))
            except ValueError as e:
                raise ValueError(
                    f"Invalid slice number: {part}"
                ) from e

    return sorted(set(slice_numbers))
