"""Regression tests for the authentication verdict and exit code."""

from __future__ import annotations

import numpy as np

import authenticate
import face_processing


def _stub_capture(monkeypatch, matched_user: str | None, confidence: float):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    aligned = np.zeros((8, 8, 3), dtype=np.uint8)
    quality = face_processing.FaceQuality(20.0, 120.0, 0.2, True, "")

    monkeypatch.setattr(
        authenticate,
        "capture_auth_image",
        lambda preview_window=True: (frame, True),
    )
    monkeypatch.setattr(
        authenticate, "align_face", lambda *_a, **_k: (aligned, quality, {})
    )
    monkeypatch.setattr(authenticate.cv2, "imwrite", lambda *_a, **_k: True)
    monkeypatch.setattr(
        authenticate,
        "verify_face",
        lambda _img, _user_id: (matched_user, confidence),
    )


def test_main_returns_false_on_wrong_identity(monkeypatch):
    _stub_capture(monkeypatch, matched_user="mallory", confidence=0.93)

    # Someone else matched the target user's enrollment: that is a failure,
    # including in the process exit code.
    assert authenticate.main(["authenticate.py", "alice"]) is False


def test_main_returns_true_on_matching_identity(monkeypatch):
    _stub_capture(monkeypatch, matched_user="alice", confidence=0.93)

    assert authenticate.main(["authenticate.py", "alice"]) is True
