__version__ = "0.0.3"


def _register_qt_metatypes():
    """Register Qt metatypes that napari's Labels signals carry across
    threads.

    Without this, PySide6 logs ``Cannot queue arguments of type
    QVector<int>`` during worker→GUI yields, and on macOS the warning
    occasionally escalates into a hard segfault. Registering the type
    is cheap, idempotent, and a no-op on qtpy versions too old to
    expose ``qRegisterMetaType``.
    """
    import contextlib

    try:
        from qtpy.QtCore import qRegisterMetaType  # type: ignore[attr-defined]
    except ImportError:
        return
    for type_name in ("QVector<int>", "QList<int>"):
        with contextlib.suppress(Exception):
            qRegisterMetaType(type_name)


_register_qt_metatypes()

from ._track_widget import TrackAnything  # noqa: E402
from ._widget import TraceAnything  # noqa: E402

__all__ = (
    "TraceAnything",
    "TrackAnything",
)
