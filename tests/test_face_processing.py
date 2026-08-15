"""Unit tests for alignment geometry and capture I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import face_processing


def test_rotated_crop_roi_uses_transformed_corners():
    bbox = (100, 80, 40, 50)
    rx, ry, rw, rh = face_processing.rotated_crop_roi(
        bbox, angle_deg=45.0, frame_wh=(400, 300), pad_ratio=0.0
    )
    cx, cy = 120.0, 105.0

    assert rw > 40
    assert rh > 50
    assert rx <= cx <= rx + rw
    assert ry <= cy <= ry + rh


def test_align_face_returns_none_without_writing(tmp_path: Path):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    aligned, quality, meta = face_processing.align_face(frame)

    assert aligned is None
    assert quality.passed is False
    assert meta is None
    assert list(tmp_path.iterdir()) == []


def test_process_frame_skips_metadata_when_disabled(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "auth_images"
    out_dir.mkdir()
    aligned = np.zeros((8, 8, 3), dtype=np.uint8)
    quality = face_processing.FaceQuality(20.0, 120.0, 0.2, True, "")
    meta = {
        "angle_deg": 0.0,
        "M": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 8,
        "roi_h": 8,
        "pad_px": 0,
        "left_eye_x": 1.0,
        "left_eye_y": 1.0,
        "right_eye_x": 2.0,
        "right_eye_y": 1.0,
    }
    monkeypatch.setattr(
        face_processing,
        "align_face",
        lambda *_args, **_kwargs: (aligned, quality, meta),
    )

    saved = face_processing.process_frame(
        np.zeros((16, 16, 3), dtype=np.uint8),
        user_id="auth",
        out_dir=str(out_dir),
        base_name="auth_capture",
        write_metadata=False,
    )

    assert saved is not None
    assert not (tmp_path / "metadata.csv").exists()
    assert not (out_dir / "metadata.csv").exists()


def test_resolve_user_id_prefers_argv():
    assert face_processing.resolve_user_id(["face_registration.py", "ks1686"]) == "ks1686"


def test_resolve_user_id_prompts_when_argv_missing():
    assert (
        face_processing.resolve_user_id(
            ["face_registration.py"], prompt=lambda _msg: "  bob  "
        )
        == "bob"
    )
