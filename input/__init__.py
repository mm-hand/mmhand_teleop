"""Configured hand input."""

from config import INPUT_SOURCE


def create_source():
    if INPUT_SOURCE == "wilor":
        from .wilor.source import WilorSource
        return WilorSource()
    if INPUT_SOURCE == "mediapipe":
        from .mediapipe.source import MediaPipeSource
        return MediaPipeSource()
    if INPUT_SOURCE == "manus":
        from .manus.calibration import ensure_calibration
        from .manus.source import ManusSource
        source = ManusSource()
        try:
            path = ensure_calibration(source.transport)
        except BaseException:
            source.close()
            raise
        source.mark_calibrated(path)
        return source
    raise ValueError(f"Unsupported INPUT_SOURCE: {INPUT_SOURCE!r}")
