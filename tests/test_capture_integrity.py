"""Saved captures must be clean frames, and failed writes must not count."""

from __future__ import annotations

import numpy as np

import face_processing
import face_registration


class _FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)

    def set(self, *_args):
        return True

    def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        pass


def test_registration_saves_frames_without_ui_overlay(monkeypatch, tmp_path):
    original = np.zeros((240, 320, 3), dtype=np.uint8)
    seen = []

    monkeypatch.setattr(face_registration, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        face_registration.cv2,
        "VideoCapture",
        lambda _i: _FakeCapture([original.copy()]),
    )
    monkeypatch.setattr(face_registration.cv2, "imshow", lambda *_a: None)
    monkeypatch.setattr(face_registration.cv2, "waitKey", lambda *_a: 0xFF)
    monkeypatch.setattr(face_registration.cv2, "destroyAllWindows", lambda: None)

    def fake_process_frame(frame_bgr, **_kwargs):
        seen.append(frame_bgr.copy())
        return None, None

    monkeypatch.setattr(face_registration, "process_frame", fake_process_frame)

    face_registration.run_guided_continuous(
        "alice", steps=[("Look ahead", 0.0)], rounds=1
    )

    assert len(seen) == 1
    # The step header, countdown, and counters are drawn for the preview only.
    assert np.array_equal(seen[0], original)


def _aligned_inputs():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    aligned = np.zeros((32, 32, 3), dtype=np.uint8)
    quality = face_processing.FaceQuality(20.0, 120.0, 0.2, True, "")
    meta = {
        "angle_deg": 0.0,
        "M": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 64,
        "roi_h": 64,
        "pad_px": 0,
        "left_eye_x": 10.0,
        "left_eye_y": 10.0,
        "right_eye_x": 50.0,
        "right_eye_y": 10.0,
    }
    return frame, aligned, quality, meta


def test_process_frame_reports_failed_image_write(monkeypatch, tmp_path):
    frame, aligned, quality, meta = _aligned_inputs()
    monkeypatch.setattr(face_processing.cv2, "imwrite", lambda *_a, **_k: False)

    out_dir = tmp_path / "alice"
    path, _q = face_processing.process_frame(
        frame,
        user_id="alice",
        out_dir=str(out_dir),
        base_name="cap",
        return_metrics=True,
        aligned=aligned,
        quality=quality,
        meta=meta,
    )

    assert path is None
    assert not (tmp_path / "metadata.csv").exists()


def test_process_frame_writes_aligned_raw_and_metadata(tmp_path):
    frame, aligned, quality, meta = _aligned_inputs()
    out_dir = tmp_path / "alice"
    path = face_processing.process_frame(
        frame,
        user_id="alice",
        out_dir=str(out_dir),
        base_name="cap",
        aligned=aligned,
        quality=quality,
        meta=meta,
    )

    assert path == str(out_dir / "cap.png")
    assert (out_dir / "cap.png").is_file()
    assert (out_dir / "raw" / "cap_raw.png").is_file()
    assert (tmp_path / "metadata.csv").is_file()
