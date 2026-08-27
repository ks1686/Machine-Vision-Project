"""Guard the MediaPipe 0.10.21 / NumPy 1.x lock so grouped bumps cannot silently regress."""

from __future__ import annotations

import cv2
import mediapipe
import numpy as np
import scipy


def _major_minor_patch(version: str) -> tuple[int, int, int]:
    parts = []
    for token in version.split(".")[:3]:
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def test_runtime_stack_stays_on_legacy_facemesh_pins():
    mp_ver = _major_minor_patch(mediapipe.__version__)
    np_major = _major_minor_patch(np.__version__)[0]
    cv_major = _major_minor_patch(cv2.__version__)[0]
    scipy_ver = _major_minor_patch(scipy.__version__)

    assert mp_ver >= (0, 10, 21)
    assert mp_ver < (0, 10, 30)
    assert np_major < 2
    assert cv_major < 5
    assert scipy_ver < (1, 18, 0)
