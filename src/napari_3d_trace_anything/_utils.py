import os
import urllib

import numpy as np
from segment_anything import sam_model_registry
from skimage.color import gray2rgb
from skimage.measure import regionprops


def load_model(model_name):
    """Load model

    Args:
        model_name (str): model name

    :return: model
    """
    model_urls = dict(default="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                      vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                      vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                      vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    model_url = model_urls[model_name]
    model_path = os.path.join(os.path.expanduser("~"), '.cache', 'napari-3d-Trace-Anything', os.path.basename(model_url))
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

    urllib.request.urlretrieve(model_url, os.path.join(os.path.expanduser("~"), '.cache', 'napari-3d-Trace-Anything', os.path.basename(model_url)))




def check_image_type(viewer, layer_name):
    image = viewer.layers[layer_name].data
    print(f'current image shape = {image.shape}')
    if len(image.shape) == 2: # Gray
        return "Not supported"
    elif len(image.shape) > 4:
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 4):
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 1): # Gray
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] > 4): # maybe stacked gray images
        return "stacked gray images"
    elif (len(image.shape) == 4)&(image.shape[-1] == 1): # maybe stacked gray images
        return "stacked gray images with channel"
    elif (len(image.shape) == 4)&(image.shape[-1] == 3): # maybe stacked RGB images
        return "stacked RGB images"
    elif (len(image.shape) == 4)&(image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 4)&(image.shape[-1] > 4):
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 3):
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


def create_box(labels):
    boxes = []
    for props in regionprops(labels):
        minr, minc, maxr, maxc = props.bbox
        box = np.array([[0, minr, minc], [0, minr, maxc], [0, maxr, maxc], [0, maxr, minc]])
        boxes.append(box)
        """
        y1 = int(coords[0][0])
        x1 = int(coords[0][1])
        y2 = int(coords[2][0])
        x2 = int(coords[2][1])
        """
    return boxes



def change_image_dtype(image):
    if image.dtype == "uint8":
        return image
    else:
        image_max = image.max()
        image_min = image.min()
        image = image - image_min # [0, N]
        image = image / (image_max - image_min) # [0, 1]
        image = image * 255 # [0, 255]
        image = image.astype(np.uint8)
        return image