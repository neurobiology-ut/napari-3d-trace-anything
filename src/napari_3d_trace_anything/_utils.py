import os
import urllib

import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from skimage.color import gray2rgb
from skimage.measure import regionprops


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
        '.cache',
        'napari-3d-Trace-Anything',
        os.path.basename(model_url))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        autodownload(model_url)
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
            '.cache',
            'napari-3d-Trace-Anything',
            os.path.basename(model_url))
            )


def check_image_type(viewer, layer_name):
    image = viewer.layers[layer_name].data
    print(f'current image shape = {image.shape}')
    if len(image.shape) == 2:  # Gray
        return "Not supported"
    elif len(image.shape) > 4:
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 4):
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 1):  # Gray
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] > 4):
        #  maybe stacked gray images
        return "stacked gray images"
    elif (len(image.shape) == 4) & (image.shape[-1] == 1):
        #  maybe stacked gray images
        return "stacked gray images with channel"
    elif (len(image.shape) == 4) & (image.shape[-1] == 3): 
        #  maybe stacked RGB images
        return "stacked RGB images"
    elif (len(image.shape) == 4) & (image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 4) & (image.shape[-1] > 4):
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 3):
        return "Not supported"
    else:
        return "Not supported"


def preprocess(image, layer_type, current_step=None):
    if layer_type == "stacked gray images":
        if current_step is not None:
            image = gray2rgb(np.array(image[current_step, :, :]))
    elif layer_type == "stacked gray images with channel":
        if current_step is not None:
            image = gray2rgb(np.array(image[current_step, :, :, 0]))
    elif layer_type == "stacked RGB images":
        if current_step is not None:
            image = np.array(image[current_step, :, :, :])
    elif layer_type == "Not supported":
        raise ValueError("image shape is not supported")
    else:
        pass
    return np.array(image)


def segment_with_sam(image, box, predictor):
    """SAMを使用して画像のセグメンテーションを行う.

    Args:
        image (np.ndarray): 入力画像
        box (np.ndarray): バウンディングボックス
        predictor (SamPredictor): SAMのpredictor

    Returns:
        np.ndarray: セグメンテーションマスク
    """
    if not isinstance(predictor, SamPredictor):
        raise ValueError("predictor must be an instance of SamPredictor")
    
    # グレースケール画像の場合、RGB形式に変換
    if len(image.shape) == 2:
        image = gray2rgb(image)
    
    # 画像をセット
    predictor.set_image(image)
    
    # ボックスプロンプトの変換
    # box形式1: [[z, y1, x1], [z, y1, x2], [z, y2, x2], [z, y2, x1]]
    # box形式2: [y1, x1, y2, x2]
    if len(box.shape) == 2:  # 形式1の場合
        input_box = np.array([
            box[0, 1],  # y1
            box[0, 2],  # x1
            box[2, 1],  # y2
            box[2, 2]   # x2
        ])
    else:  # 形式2の場合
        input_box = box
    
    # マスクの生成
    masks, _, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=False
    )
    
    return masks[0]


def create_box(labels):
    boxes = []
    for props in regionprops(labels):
        minr, minc, maxr, maxc = props.bbox
        box = np.array([
            [0, minr, minc],
            [0, minr, maxc],
            [0, maxr, maxc],
            [0, maxr, minc]]
            )
        boxes.append(box)
        """
        y1 = int(coords[0][0])
        x1 = int(coords[0][1])
        y2 = int(coords[2][0])
        x2 = int(coords[2][1])
        """
    return boxes
