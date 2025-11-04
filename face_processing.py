"""
face_processing.py
------------------
Utilities for face detection, quality assessment, alignment, and saving.

Tunable parameters:
- MIN_BLUR_VAR: laptop webcams often yield 5–20; raise for stricter blur checks.
- BRIGHTNESS_RANGE: widen for dim rooms, tighten for controlled lighting.
- MIN_BBOX_RATIO: face area / frame area; lower if users sit far from camera.
- OUTPUT_SIZE: final aligned crop size (H, W).
"""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np
import pandas as pd

# -------- Quality / alignment parameters (tune here as needed) --------
MIN_BLUR_VAR = 10.0  # variance of Laplacian on sharpened ROI
MIN_BBOX_RATIO = 0.04  # min face area / frame area
BRIGHTNESS_RANGE = (40, 235)
OUTPUT_SIZE = (224, 224)  # (H, W)

# MediaPipe indices for outer eye corners.
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# Persistent FaceMesh
_FACE_MESH = None


def get_face_mesh():
    """Return a persistent MediaPipe FaceMesh instance (single-face)."""
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH


@atexit.register
def _close_face_mesh():
    """Close FaceMesh on interpreter exit to release resources."""
    global _FACE_MESH
    if _FACE_MESH is not None:
        try:
            _FACE_MESH.close()
        except Exception:
            pass


@dataclass
class FaceQuality:
    blur_var: float
    brightness: float
    bbox_ratio: float
    passed: bool
    reasons: str = ""


# ---- Metrics helpers ----


def variance_of_laplacian(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def brightness_mean(img_gray: np.ndarray) -> float:
    return float(np.mean(img_gray))


def _unsharp_mask(gray: np.ndarray, amount: float = 1.5, radius: int = 1) -> np.ndarray:
    """Simple unsharp mask to boost edges; helps low-res laptop webcams."""
    blurred = cv2.GaussianBlur(gray, (radius * 2 + 1, radius * 2 + 1), 0)
    sharp = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _clahe(
    gray: np.ndarray, clip: float = 2.0, tiles: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Local contrast normalization helps low-light laptop cameras."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return clahe.apply(gray)


# ---- Core ops ----


def detect_face_and_landmarks(
    frame_bgr: np.ndarray,
) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Return (bbox_xywh, landmarks_xy) in pixel coordinates; None if not found."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fm = get_face_mesh()
    res = cast(Any, fm.process(rgb))
    if not getattr(res, "multi_face_landmarks", None):
        return None

    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    landmarks_xy = np.stack([xs, ys], axis=1)  # (468, 2)

    x_min, y_min = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(ys.min(), 0, h - 1))
    x_max, y_max = int(np.clip(xs.max(), 0, w - 1)), int(np.clip(ys.max(), 0, h - 1))
    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    return bbox, landmarks_xy


def eye_aligned_face(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Rotate so the eye line is horizontal, crop a padded ROI, then resize."""
    x, y, w, h = bbox_xywh
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (x + w // 2, y + h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        frame_bgr,
        M,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    pad = int(0.2 * max(w, h))  # include forehead/chin
    rx, ry = max(0, x - pad), max(0, y - pad)
    rw = min(rotated.shape[1] - rx, w + 2 * pad)
    rh = min(rotated.shape[0] - ry, h + 2 * pad)
    crop = rotated[ry : ry + rh, rx : rx + rw]

    return cv2.resize(
        crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
    )


def assess_quality(
    frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]
) -> FaceQuality:
    """Compute blur (Laplacian var), brightness, and face-box ratio; return reasons if rejected."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox_xywh
    face = frame_bgr[y : y + bh, x : x + bw]

    gray0 = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_eq = _clahe(gray0)
    gray = _unsharp_mask(gray_eq, amount=1.5, radius=1)

    blur = variance_of_laplacian(gray)
    bright = brightness_mean(gray0)
    bbox_ratio = (bw * bh) / float(w * h)

    reasons = []
    if blur < MIN_BLUR_VAR:
        reasons.append(f"blur {blur:.0f} < {MIN_BLUR_VAR}")
    if bright < BRIGHTNESS_RANGE[0] or bright > BRIGHTNESS_RANGE[1]:
        reasons.append(
            f"brightness {bright:.0f} ∉ [{BRIGHTNESS_RANGE[0]}, {BRIGHTNESS_RANGE[1]}]"
        )
    if bbox_ratio < MIN_BBOX_RATIO:
        reasons.append(f"bbox {bbox_ratio:.3f} < {MIN_BBOX_RATIO}")

    return FaceQuality(blur, bright, bbox_ratio, len(reasons) == 0, "; ".join(reasons))


# ---- Metadata ----


def ensure_csv(path: str):
    if not os.path.exists(path):
        pd.DataFrame(
            columns=[
                "timestamp",
                "user_id",
                "file",
                "blur_var",
                "brightness",
                "bbox_ratio",
                "frame_w",
                "frame_h",
                "aligned_h",
                "aligned_w",
            ]
        ).to_csv(path, index=False)


def append_metadata(csv_path: str, row: dict):
    ensure_csv(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)


# ---- Public API ----


def process_frame(
    frame_bgr: np.ndarray,
    user_id: str,
    out_dir: str,
    base_name: str,
    return_metrics: bool = False,
) -> Any:
    """Detect, quality-check, align, and save one face frame. Optionally return metrics."""
    detection = detect_face_and_landmarks(frame_bgr)
    if detection is None:
        if return_metrics:
            return None, FaceQuality(0.0, 0.0, 0.0, False, "no face detected")
        return None

    bbox, landmarks = detection
    q = assess_quality(frame_bgr, bbox)
    if not q.passed:
        if return_metrics:
            return None, q
        return None

    aligned = eye_aligned_face(frame_bgr, landmarks, bbox, OUTPUT_SIZE)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{base_name}.jpg")
    cv2.imwrite(fpath, aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

    h, w = frame_bgr.shape[:2]
    append_metadata(
        os.path.join(os.path.dirname(out_dir), "metadata.csv"),
        {
            "timestamp": int(time.time()),
            "user_id": user_id,
            "file": os.path.relpath(fpath, os.path.dirname(out_dir)),
            "blur_var": q.blur_var,
            "brightness": q.brightness,
            "bbox_ratio": q.bbox_ratio,
            "frame_w": w,
            "frame_h": h,
            "aligned_h": OUTPUT_SIZE[0],
            "aligned_w": OUTPUT_SIZE[1],
        },
    )

    if return_metrics:
        return fpath, q
    return fpath
