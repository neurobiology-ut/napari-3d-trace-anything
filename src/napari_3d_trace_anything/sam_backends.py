# Remote backend adapted from napari-gc-analysis (Apache-2.0)
# https://github.com/neurobiology-ut/napari-gc-analysis

import io
import json
import pickle
from typing import Hashable, Optional, Protocol

import numpy as np
import requests
from skimage.color import gray2rgb


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 for transport to the SAM server."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float64)
    max_val = float(img.max()) if img.size else 0.0
    if max_val <= 0 or not np.isfinite(max_val):
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip(img / max_val * 255.0, 0, 255).astype(np.uint8)


class SAMBackend(Protocol):
    """A two-stage SAM inference interface.

    ``prepare`` makes the backend ready for a given frame; ``segment``
    runs the network with a single box and returns the mask. Splitting
    the two stages lets callers feed multiple boxes per frame without
    re-uploading the frame, and lets the local backend keep the
    expensive ``set_image`` work cached by frame identity instead of
    array equality.
    """

    def prepare(self, image: np.ndarray, image_id: Hashable) -> None:
        ...

    def segment(self, box_xyxy: np.ndarray) -> np.ndarray:
        ...


class LocalSAMBackend:
    """Run SAM inside the napari process via a ``SamPredictor``.

    ``set_image`` is cached by ``image_id`` (the caller-supplied
    identity — typically a frame index) so repeated boxes on the same
    frame skip the ~hundred-millisecond image encode.
    """

    def __init__(self, predictor):
        if not (
            hasattr(predictor, "set_image") and hasattr(predictor, "predict")
        ):
            raise ValueError(
                "predictor must implement set_image and predict"
            )
        self._predictor = predictor
        self._loaded_id: Optional[Hashable] = None

    def prepare(self, image: np.ndarray, image_id: Hashable) -> None:
        if image_id == self._loaded_id:
            return
        if image.ndim == 2:
            image = gray2rgb(image)
        self._predictor.set_image(image)
        self._loaded_id = image_id

    def segment(self, box_xyxy: np.ndarray) -> np.ndarray:
        if self._loaded_id is None:
            raise RuntimeError("prepare() must be called before segment()")
        box = np.asarray(box_xyxy, dtype=np.float32)
        masks, _scores, _logits = self._predictor.predict(
            box=box[None, :], multimask_output=False
        )
        return masks[0].astype(bool)


class RemoteSAMBackend:
    """POST a frame + single box to a remote SAM server (pickled).

    Wire format matches napari-gc-analysis's server: uint8 image is
    pickled into a multipart file under ``numpy_data``, box coordinates
    travel as JSON, response body is a pickled boolean mask.
    """

    def __init__(self, url: str, model: str = "sam", session=None):
        if not url:
            raise ValueError("RemoteSAMBackend requires a non-empty url")
        self.url = url
        self.model = model
        # Session is injectable so tests can stub network IO without
        # monkey-patching the module-level ``requests.post``.
        self._session = session if session is not None else requests.Session()
        self._image_uint8: Optional[np.ndarray] = None
        self._image_id: Optional[Hashable] = None

    def prepare(self, image: np.ndarray, image_id: Hashable) -> None:
        if image_id == self._image_id:
            return
        self._image_uint8 = _to_uint8(image)
        self._image_id = image_id

    def segment(self, box_xyxy: np.ndarray) -> np.ndarray:
        if self._image_uint8 is None:
            raise RuntimeError("prepare() must be called before segment()")
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        pickle_data = pickle.dumps(self._image_uint8)
        coords_json = json.dumps({"coords": [[x1, y1, x2, y2]]})
        response = self._session.post(
            self.url,
            files={"numpy_data": io.BytesIO(pickle_data)},
            data={"coords": coords_json, "model": self.model},
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type != "application/octet-stream":
            raise RuntimeError(
                f"server returned non-pickle Content-Type: {content_type!r}"
            )
        mask = pickle.loads(response.content)
        return np.asarray(mask).astype(bool)
