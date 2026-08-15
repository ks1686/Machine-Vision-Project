"""Unit tests for comparison scoring, liveness, and model discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import compare_faces


def _blank_landmarks(n: int = 478) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float32)


def _frontal() -> np.ndarray:
    lm = _blank_landmarks()
    lm[1] = [0.0, 0.1, 0.2]
    lm[152] = [0.0, 0.5, 0.1]
    lm[33] = [-0.3, 0.0, 0.15]
    lm[263] = [0.3, 0.0, 0.15]
    lm[61] = [-0.15, 0.35, 0.1]
    lm[291] = [0.15, 0.35, 0.1]
    return lm


def _yawed() -> np.ndarray:
    lm = _frontal()
    lm[33] = [-0.05, 0.0, 0.15]
    lm[263] = [0.6, 0.0, 0.15]
    return lm


def test_identical_features_score_near_one():
    features = _frontal()
    metrics = compare_faces.compare_features(features, features.copy())
    assert metrics["score"] == pytest.approx(1.0)


def test_dissimilar_features_score_near_zero():
    features = _frontal()
    other = features + 10.0
    metrics = compare_faces.compare_features(features, other)
    assert metrics["score"] == pytest.approx(0.0)
    assert metrics["score"] < 0.35


def test_large_pose_diff_rejects():
    metrics = compare_faces.compare_features(_frontal(), _yawed())
    assert metrics["pose_diff"] > 30.0
    assert metrics["score"] == pytest.approx(0.0)


def test_verify_face_rejects_low_depth_variance(monkeypatch):
    flat = (_frontal(), 0.0001)
    loaded = {"called": False}
    monkeypatch.setattr(compare_faces, "get_face_features", lambda _img: flat)

    def fake_load(*_args, **_kwargs):
        loaded["called"] = True
        return ["alice"]

    monkeypatch.setattr(compare_faces, "load_registered_models", fake_load)

    matched, score = compare_faces.verify_face(np.zeros((512, 512, 3), np.uint8))

    assert matched is None
    assert score == pytest.approx(0.0)
    assert loaded["called"] is False


def test_load_registered_models_includes_avgface_without_obj(tmp_path: Path):
    models_dir = tmp_path / "models"
    faces_dir = tmp_path / "registered_faces"
    models_dir.mkdir()
    user_dir = faces_dir / "alice"
    user_dir.mkdir(parents=True)
    (user_dir / "alice_avgface.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    users = compare_faces.load_registered_models(
        models_dir=str(models_dir), faces_dir=str(faces_dir)
    )

    assert users == ["alice"]


def test_model_feature_cache_reads_image_once(monkeypatch, tmp_path: Path):
    img_path = tmp_path / "bob_avgface.png"
    img_path.write_bytes(b"not-a-real-png")
    calls = {"n": 0}
    cached = (_frontal(), 0.02)

    def fake_features(_img):
        calls["n"] += 1
        return cached

    monkeypatch.setattr(compare_faces, "get_face_features", fake_features)
    monkeypatch.setattr(compare_faces.cv2, "imread", lambda _p: np.zeros((4, 4, 3), np.uint8))
    compare_faces.clear_feature_cache()

    first = compare_faces.get_cached_model_features("bob", str(img_path))
    second = compare_faces.get_cached_model_features("bob", str(img_path))

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first[0], cached[0])
    assert calls["n"] == 1


def test_compare_with_registration_frames_removed():
    assert not hasattr(compare_faces, "compare_with_registration_frames")
