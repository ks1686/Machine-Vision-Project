"""User ID validation, private permissions, and safe model discovery."""

from __future__ import annotations

import os
import stat
import sys

import cv2
import numpy as np
import pytest

import authenticate
import compare_faces
import face_processing
import face_registration
import secure_storage

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")


@pytest.mark.parametrize(
    "raw,expected",
    [("alice", "alice"), (" Bob Smith ", "Bob_Smith"), ("u-1.2_x", "u-1.2_x")],
)
def test_valid_user_ids(raw, expected):
    assert secure_storage.validate_user_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "../etc", "..", "a/b", "a\\b", "/abs", ".hidden", "a..b", "x" * 65, "nul\x00"],
)
def test_rejects_unsafe_user_ids(raw):
    with pytest.raises(ValueError):
        secure_storage.validate_user_id(raw)


def test_registration_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(face_registration, "OUTPUT_DIR", str(tmp_path / "faces"))
    opened = []
    monkeypatch.setattr(
        face_registration.cv2, "VideoCapture", lambda i: opened.append(i)
    )
    with pytest.raises(ValueError):
        face_registration.run_guided_continuous("../../outside")
    assert opened == []
    assert not (tmp_path / "outside").exists()


def test_authenticate_rejects_path_traversal(monkeypatch):
    called = []
    monkeypatch.setattr(
        authenticate, "capture_auth_image", lambda **_k: called.append(1)
    )
    assert authenticate.main(["authenticate.py", "../alice"]) is False
    assert called == []


def test_verify_face_rejects_unsafe_target(monkeypatch):
    lm = np.random.default_rng(0).random((478, 3)).astype(np.float32)
    monkeypatch.setattr(compare_faces, "get_face_features", lambda _img: (lm, 0.5))
    monkeypatch.setattr(
        compare_faces,
        "get_cached_model_features",
        lambda *_a: pytest.fail("must not read files for an unsafe user ID"),
    )
    assert compare_faces.verify_face(np.zeros((8, 8, 3), np.uint8), "../x") == (
        None,
        0.0,
    )


def test_load_registered_models_skips_unsafe_names(tmp_path):
    faces = tmp_path / "faces"
    (faces / "alice").mkdir(parents=True)
    (faces / "alice" / "alice_avgface.png").write_bytes(b"x")
    (faces / ".hidden").mkdir()
    (faces / ".hidden" / ".hidden_avgface.png").write_bytes(b"x")
    assert compare_faces.load_registered_models(str(tmp_path / "none"), str(faces)) == [
        "alice"
    ]


@posix_only
def test_process_frame_writes_owner_only_files(tmp_path):
    old = os.umask(0o022)
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        quality = face_processing.FaceQuality(20.0, 120.0, 0.2, True, "")
        meta = {
            k: 0 for k in ("angle_deg", "roi_x", "roi_y", "roi_w", "roi_h", "pad_px")
        }
        meta.update(
            M=[1.0, 0, 0, 0, 1.0, 0],
            left_eye_x=0.0,
            left_eye_y=0.0,
            right_eye_x=1.0,
            right_eye_y=0.0,
        )
        out_dir = tmp_path / "faces" / "alice"
        path = face_processing.process_frame(
            frame,
            user_id="alice",
            out_dir=str(out_dir),
            base_name="cap",
            aligned=frame[:32, :32],
            quality=quality,
            meta=meta,
        )
    finally:
        os.umask(old)

    def mode(p):
        return stat.S_IMODE(os.stat(p).st_mode)

    assert mode(path) == 0o600
    assert mode(out_dir / "raw" / "cap_raw.png") == 0o600
    assert mode(tmp_path / "faces" / "metadata.csv") == 0o600
    assert mode(out_dir) == 0o700
    assert mode(out_dir / "raw") == 0o700


def test_feature_cache_refreshes_when_avg_face_changes(monkeypatch, tmp_path):
    compare_faces.clear_feature_cache()
    img_path = tmp_path / "bob_avgface.png"
    cv2.imwrite(str(img_path), np.zeros((8, 8, 3), np.uint8))
    calls = []

    def fake_features(_img):
        calls.append(1)
        return np.zeros((478, 3), np.float32), float(len(calls))

    monkeypatch.setattr(compare_faces, "get_face_features", fake_features)
    first = compare_faces.get_cached_model_features("bob", str(img_path))
    st = os.stat(img_path)
    os.utime(img_path, (st.st_atime, st.st_mtime + 10))
    second = compare_faces.get_cached_model_features("bob", str(img_path))
    assert first[1] == 1.0 and second[1] == 2.0
    compare_faces.clear_feature_cache()
