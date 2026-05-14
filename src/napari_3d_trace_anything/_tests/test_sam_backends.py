import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_3d_trace_anything import _utils
from napari_3d_trace_anything.sam_backends import (
    LocalSAMBackend,
    RemoteSAMBackend,
)


class _FakePredictor:
    """Records set_image / predict calls for assertions."""

    def __init__(self, mask=None):
        self.set_image_calls = []
        self.predict_calls = []
        h, w = 100, 100
        self._mask = (
            mask if mask is not None else np.ones((h, w), dtype=bool)
        )

    def set_image(self, image):
        self.set_image_calls.append(image)

    def predict(self, box, multimask_output):
        self.predict_calls.append(box.copy())
        return np.array([self._mask]), np.array([1.0]), None


# ---------- LocalSAMBackend ----------


def test_local_backend_prepare_caches_by_image_id():
    pred = _FakePredictor()
    backend = LocalSAMBackend(pred)

    image = np.zeros((100, 100), dtype=np.uint8)
    backend.prepare(image, image_id=0)
    backend.prepare(image, image_id=0)
    backend.prepare(image, image_id=0)

    assert len(pred.set_image_calls) == 1


def test_local_backend_prepare_reloads_on_new_id():
    pred = _FakePredictor()
    backend = LocalSAMBackend(pred)

    image = np.zeros((100, 100), dtype=np.uint8)
    backend.prepare(image, image_id=0)
    backend.prepare(image, image_id=1)

    assert len(pred.set_image_calls) == 2


def test_local_backend_converts_2d_grayscale_to_rgb():
    pred = _FakePredictor()
    backend = LocalSAMBackend(pred)

    image = np.zeros((100, 100), dtype=np.uint8)
    backend.prepare(image, image_id=0)

    # gray2rgb returns (H, W, 3)
    assert pred.set_image_calls[0].ndim == 3
    assert pred.set_image_calls[0].shape[-1] == 3


def test_local_backend_segment_returns_bool_mask():
    pred = _FakePredictor(mask=np.ones((100, 100), dtype=np.uint8))
    backend = LocalSAMBackend(pred)
    image = np.zeros((100, 100), dtype=np.uint8)
    backend.prepare(image, image_id=0)

    mask = backend.segment(np.array([10, 10, 50, 50]))

    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)
    assert pred.predict_calls[0].shape == (1, 4)


def test_local_backend_segment_before_prepare_raises():
    backend = LocalSAMBackend(_FakePredictor())
    with pytest.raises(RuntimeError, match="prepare"):
        backend.segment(np.array([0, 0, 10, 10]))


def test_local_backend_rejects_non_predictor():
    with pytest.raises(ValueError, match="set_image"):
        LocalSAMBackend(object())


# ---------- RemoteSAMBackend ----------


def _make_session_returning_mask(mask: np.ndarray) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/octet-stream"}
    response.content = pickle.dumps(mask)
    response.raise_for_status.return_value = None
    session = MagicMock()
    session.post.return_value = response
    return session


def test_remote_backend_empty_url_raises():
    with pytest.raises(ValueError, match="non-empty url"):
        RemoteSAMBackend(url="")


def test_remote_backend_uint16_to_uint8():
    """uint16 input must be normalized to uint8 before posting."""
    expected_mask = np.ones((100, 100), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)

    backend = RemoteSAMBackend(
        url="http://stub", model="sam", session=session
    )
    image = np.full((100, 100), 12345, dtype=np.uint16)
    backend.prepare(image, image_id=0)
    mask = backend.segment(np.array([10, 10, 50, 50]))

    # Inspect what was actually posted.
    posted_pickle = session.post.call_args.kwargs["files"]["numpy_data"]
    arr = pickle.loads(posted_pickle.getvalue())
    assert arr.dtype == np.uint8
    assert mask.dtype == np.bool_


def test_remote_backend_zero_image_no_div_by_zero():
    expected_mask = np.zeros((100, 100), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)

    backend = RemoteSAMBackend(url="http://stub", session=session)
    image = np.zeros((100, 100, 3), dtype=np.uint16)
    backend.prepare(image, image_id=0)
    mask = backend.segment(np.array([10, 10, 30, 30]))

    posted_pickle = session.post.call_args.kwargs["files"]["numpy_data"]
    arr = pickle.loads(posted_pickle.getvalue())
    assert arr.dtype == np.uint8
    assert mask.dtype == np.bool_


def test_remote_backend_prepare_caches_by_image_id():
    expected_mask = np.ones((100, 100), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)

    backend = RemoteSAMBackend(url="http://stub", session=session)
    image_a = np.zeros((100, 100), dtype=np.uint16)
    image_b = np.ones((100, 100), dtype=np.uint16)

    backend.prepare(image_a, image_id=0)
    cached = backend._image_uint8
    backend.prepare(image_a, image_id=0)
    # Same id, no reconversion.
    assert backend._image_uint8 is cached

    backend.prepare(image_b, image_id=1)
    assert backend._image_uint8 is not cached


def test_remote_backend_segment_before_prepare_raises():
    session = _make_session_returning_mask(
        np.zeros((10, 10), dtype=np.uint8)
    )
    backend = RemoteSAMBackend(url="http://stub", session=session)
    with pytest.raises(RuntimeError, match="prepare"):
        backend.segment(np.array([0, 0, 5, 5]))


def test_remote_backend_non_pickle_response_raises():
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "text/html"}
    response.content = b"<html>down</html>"
    response.raise_for_status.return_value = None
    session = MagicMock()
    session.post.return_value = response

    backend = RemoteSAMBackend(url="http://stub", session=session)
    image = np.zeros((100, 100), dtype=np.uint8)
    backend.prepare(image, image_id=0)
    with pytest.raises(RuntimeError, match="non-pickle"):
        backend.segment(np.array([10, 10, 30, 30]))


# ---------- get_sam_model cache ----------


def test_get_sam_model_caches_weights(monkeypatch):
    """Same model name loads weights once; subsequent calls reuse them."""
    monkeypatch.setattr(_utils, "_sam_model_cache", {})

    load_calls = []

    class _FakeSAM:
        def to(self, device):
            self.device = device
            return self

    def fake_load(name):
        load_calls.append(name)
        return _FakeSAM()

    monkeypatch.setattr(_utils, "load_model", fake_load)

    a = _utils.get_sam_model("vit_h")
    b = _utils.get_sam_model("vit_h")
    assert a is b
    assert load_calls == ["vit_h"]


def test_get_sam_model_different_names_load_separately(monkeypatch):
    monkeypatch.setattr(_utils, "_sam_model_cache", {})

    load_calls = []

    class _FakeSAM:
        def to(self, device):
            return self

    def fake_load(name):
        load_calls.append(name)
        return _FakeSAM()

    monkeypatch.setattr(_utils, "load_model", fake_load)

    _utils.get_sam_model("vit_h")
    _utils.get_sam_model("vit_b")
    assert load_calls == ["vit_h", "vit_b"]


def test_get_sam_model_moves_to_device(monkeypatch):
    monkeypatch.setattr(_utils, "_sam_model_cache", {})

    class _FakeSAM:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(_utils, "load_model", lambda name: _FakeSAM())

    sam = _utils.get_sam_model("vit_h", device="cpu")
    assert sam.device == "cpu"
