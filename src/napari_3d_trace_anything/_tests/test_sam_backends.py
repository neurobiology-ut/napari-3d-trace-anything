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
        self._mask = mask if mask is not None else np.ones((h, w), dtype=bool)

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

    backend = RemoteSAMBackend(url="http://stub", model="sam", session=session)
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
    session = _make_session_returning_mask(np.zeros((10, 10), dtype=np.uint8))
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


def test_get_sam_model_rejects_device_conflict(monkeypatch):
    """Asking for the same model on a different device must raise rather
    than silently migrating the cached weights and breaking the earlier
    predictor."""
    monkeypatch.setattr(_utils, "_sam_model_cache", {})

    class _FakeSAM:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(_utils, "load_model", lambda name: _FakeSAM())

    _utils.get_sam_model("vit_h", device="cuda")
    # Same device: fine.
    _utils.get_sam_model("vit_h", device="cuda")
    # No device specified: returns the cached instance without moving.
    _utils.get_sam_model("vit_h")
    # Different device: refuses to migrate.
    with pytest.raises(RuntimeError, match="already loaded on"):
        _utils.get_sam_model("vit_h", device="cpu")


def test_get_sam_model_late_device_assignment(monkeypatch):
    """A first caller without device, then a second with one, should
    move the cached weights (no predictor in flight yet) and remember
    that device for the conflict check."""
    monkeypatch.setattr(_utils, "_sam_model_cache", {})

    class _FakeSAM:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(_utils, "load_model", lambda name: _FakeSAM())

    sam = _utils.get_sam_model("vit_h")
    assert sam.device is None
    _utils.get_sam_model("vit_h", device="cpu")
    assert sam.device == "cpu"
    with pytest.raises(RuntimeError):
        _utils.get_sam_model("vit_h", device="cuda")


def test_remote_backend_default_timeout_passed_to_session():
    expected_mask = np.ones((10, 10), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)
    backend = RemoteSAMBackend(url="http://stub", session=session)
    backend.prepare(np.zeros((10, 10), dtype=np.uint8), image_id=0)
    backend.segment(np.array([0, 0, 5, 5]))

    timeout = session.post.call_args.kwargs.get("timeout")
    assert timeout == RemoteSAMBackend.DEFAULT_TIMEOUT_SECONDS


def test_remote_backend_custom_timeout_passed_to_session():
    expected_mask = np.ones((10, 10), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)
    backend = RemoteSAMBackend(url="http://stub", session=session, timeout=5.0)
    backend.prepare(np.zeros((10, 10), dtype=np.uint8), image_id=0)
    backend.segment(np.array([0, 0, 5, 5]))

    assert session.post.call_args.kwargs["timeout"] == 5.0


def test_remote_backend_timeout_none_disables_it():
    """Passing None must propagate so callers can opt out of the bound."""
    expected_mask = np.ones((10, 10), dtype=np.uint8)
    session = _make_session_returning_mask(expected_mask)
    backend = RemoteSAMBackend(
        url="http://stub", session=session, timeout=None
    )
    backend.prepare(np.zeros((10, 10), dtype=np.uint8), image_id=0)
    backend.segment(np.array([0, 0, 5, 5]))

    assert session.post.call_args.kwargs["timeout"] is None


# ---------- autodownload / load_model robustness (PR#10) ----------


def _make_zip_bytes():
    """Build a tiny but valid zip archive byte string."""
    import io
    import zipfile as _zf

    buf = io.BytesIO()
    with _zf.ZipFile(buf, "w") as z:
        z.writestr("placeholder.bin", b"x")
    return buf.getvalue()


def _fake_requests_get(payload_bytes, content_length=None):
    """Return a fake `requests.get` context manager yielding `payload_bytes`.

    Setting ``content_length`` to something other than ``len(payload)``
    simulates a truncated download.
    """
    if content_length is None:
        content_length = len(payload_bytes)

    class _Resp:
        headers = {"Content-Length": str(content_length)}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            for i in range(0, len(payload_bytes), chunk_size):
                yield payload_bytes[i : i + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_get(_url, stream=False, timeout=None):
        return _Resp()

    return fake_get


def test_autodownload_atomic_success(monkeypatch, tmp_path):
    """A valid zip download lands at the final path with no `.part` left."""
    monkeypatch.setattr(
        _utils,
        "_model_path_for_url",
        lambda url: str(tmp_path / "checkpoint.pth"),
    )
    payload = _make_zip_bytes()
    monkeypatch.setattr(
        _utils, "requests", MagicMock(get=_fake_requests_get(payload))
    )

    _utils.autodownload("http://stub/checkpoint.pth")

    final = tmp_path / "checkpoint.pth"
    assert final.exists()
    assert final.read_bytes() == payload
    assert not (tmp_path / "checkpoint.pth.part").exists()


def test_autodownload_truncated_raises_and_cleans_part(monkeypatch, tmp_path):
    """Content-Length mismatch → OSError, no `.part` leftover."""
    monkeypatch.setattr(
        _utils,
        "_model_path_for_url",
        lambda url: str(tmp_path / "checkpoint.pth"),
    )
    payload = _make_zip_bytes()
    monkeypatch.setattr(
        _utils,
        "requests",
        MagicMock(
            get=_fake_requests_get(payload, content_length=len(payload) + 999)
        ),
    )

    with pytest.raises(OSError, match="truncated"):
        _utils.autodownload("http://stub/checkpoint.pth")

    assert not (tmp_path / "checkpoint.pth").exists()
    assert not (tmp_path / "checkpoint.pth.part").exists()


def test_autodownload_non_zip_raises_and_cleans_part(monkeypatch, tmp_path):
    """Server returns non-zip bytes → OSError, no `.part` leftover."""
    monkeypatch.setattr(
        _utils,
        "_model_path_for_url",
        lambda url: str(tmp_path / "checkpoint.pth"),
    )
    junk = b"<html>404 not found</html>"
    monkeypatch.setattr(
        _utils, "requests", MagicMock(get=_fake_requests_get(junk))
    )

    with pytest.raises(OSError, match="not a valid zip"):
        _utils.autodownload("http://stub/checkpoint.pth")

    assert not (tmp_path / "checkpoint.pth").exists()
    assert not (tmp_path / "checkpoint.pth.part").exists()


def test_autodownload_skips_when_final_already_valid(monkeypatch, tmp_path):
    """If another thread finished the DL first, autodownload exits early."""
    final = tmp_path / "checkpoint.pth"
    final.write_bytes(_make_zip_bytes())
    monkeypatch.setattr(_utils, "_model_path_for_url", lambda url: str(final))

    # If the fast path is missed, this fake will fail because there's no
    # `Content-Length` header configured beyond the early-exit check.
    called = {"n": 0}

    def fake_get(*a, **kw):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(_utils, "requests", MagicMock(get=fake_get))

    _utils.autodownload("http://stub/checkpoint.pth")
    assert called["n"] == 0


def test_load_model_recovers_from_corrupt_checkpoint(monkeypatch, tmp_path):
    """A pre-existing corrupt `.pth` is deleted + re-downloaded once."""
    final = tmp_path / "sam_vit_b_01ec64.pth"
    final.write_bytes(b"definitely-not-a-zip")
    monkeypatch.setattr(_utils, "_model_path_for_url", lambda url: str(final))

    fake_registry = {"vit_b": MagicMock(return_value="SAM-instance")}
    monkeypatch.setattr(
        _utils, "_MODEL_URLS", {"vit_b": "http://stub/sam_vit_b_01ec64.pth"}
    )
    import sys

    sam_anything_mod = MagicMock()
    sam_anything_mod.sam_model_registry = fake_registry
    monkeypatch.setitem(sys.modules, "segment_anything", sam_anything_mod)

    download_calls = {"n": 0}

    def fake_autodownload(url):
        download_calls["n"] += 1
        final.write_bytes(_make_zip_bytes())

    monkeypatch.setattr(_utils, "autodownload", fake_autodownload)

    result = _utils.load_model("vit_b")
    assert result == "SAM-instance"
    assert download_calls["n"] == 1  # corrupt → one re-download


def test_load_model_double_corrupt_raises_clear_error(monkeypatch, tmp_path):
    """If re-download still produces a non-zip, raise a clear RuntimeError."""
    final = tmp_path / "sam_vit_b_01ec64.pth"
    final.write_bytes(b"junk-1")
    monkeypatch.setattr(_utils, "_model_path_for_url", lambda url: str(final))
    monkeypatch.setattr(
        _utils, "_MODEL_URLS", {"vit_b": "http://stub/sam_vit_b_01ec64.pth"}
    )

    def fake_autodownload(url):
        # Pretend the server keeps serving garbage.
        final.write_bytes(b"junk-2")

    monkeypatch.setattr(_utils, "autodownload", fake_autodownload)

    with pytest.raises(RuntimeError, match="still corrupt"):
        _utils.load_model("vit_b")


def test_get_sam_model_default_and_vit_h_share_instance(monkeypatch):
    """`default` and `vit_h` resolve to the same checkpoint and SAM instance."""
    monkeypatch.setattr(_utils, "_sam_model_cache", {})
    monkeypatch.setattr(_utils, "_checkpoint_locks", {})

    class _FakeSAM:
        def to(self, device):
            return self

    load_calls = []

    def fake_load(name):
        load_calls.append(name)
        return _FakeSAM()

    monkeypatch.setattr(_utils, "load_model", fake_load)

    a = _utils.get_sam_model("default")
    b = _utils.get_sam_model("vit_h")
    assert a is b
    assert load_calls == ["default"]  # only one underlying load


def test_get_sam_model_serializes_concurrent_same_checkpoint(monkeypatch):
    """Two threads asking for the same checkpoint serialize: load runs once.

    Thread A enters first and sleeps inside the lock; thread B is
    started while A is still loading, must block on the lock, and on
    acquiring it should find the cache populated → skips load_model
    entirely.
    """
    import threading as _t
    import time

    monkeypatch.setattr(_utils, "_sam_model_cache", {})
    monkeypatch.setattr(_utils, "_checkpoint_locks", {})

    class _FakeSAM:
        def to(self, device):
            return self

    inside_load = _t.Event()
    load_calls = []

    def fake_load(name):
        load_calls.append(name)
        inside_load.set()
        time.sleep(0.15)
        return _FakeSAM()

    monkeypatch.setattr(_utils, "load_model", fake_load)

    results = {}

    def run(key, name):
        results[key] = _utils.get_sam_model(name)

    t_a = _t.Thread(target=run, args=("a", "default"))
    t_a.start()
    # Wait until A is INSIDE fake_load (lock held).
    assert inside_load.wait(timeout=2.0)
    t_b = _t.Thread(target=run, args=("b", "vit_h"))
    t_b.start()
    t_a.join(timeout=5.0)
    t_b.join(timeout=5.0)

    # Only one underlying load — the second caller got the cache hit.
    assert len(load_calls) == 1
    assert results["a"] is results["b"]


def test_get_sam_model_parallel_different_checkpoints(monkeypatch):
    """vit_h + vit_b are independent checkpoints → parallel allowed."""
    import threading as _t

    monkeypatch.setattr(_utils, "_sam_model_cache", {})
    monkeypatch.setattr(_utils, "_checkpoint_locks", {})

    class _FakeSAM:
        def to(self, device):
            return self

    barrier = _t.Barrier(2)
    enter_times = []
    import time

    def fake_load(name):
        enter_times.append(time.monotonic())
        barrier.wait(timeout=2.0)
        time.sleep(0.1)
        return _FakeSAM()

    monkeypatch.setattr(_utils, "load_model", fake_load)

    threads = [
        _t.Thread(target=_utils.get_sam_model, args=("vit_h",)),
        _t.Thread(target=_utils.get_sam_model, args=("vit_b",)),
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    wall = time.monotonic() - t0

    # If the lock serialized them, wall would be ~0.2s. Parallel allows ~0.1s.
    assert wall < 0.18, f"different checkpoints should be parallel; got {wall}"
    assert len(enter_times) == 2
