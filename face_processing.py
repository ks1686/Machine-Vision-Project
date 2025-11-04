"""
face_processing.py

Processes faces in images: detection, quality assessment.
"""

from __future__ import annotations
import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import time
import os
import atexit
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Any, cast

# Change as needed when testing
MIN_BLUR_VAR = (
    10.0  # Variance of Laplacian on pre-sharpened ROI; tuned for laptop webcams
)
MIN_BBOX_RATIO = 0.04  # min face area / frame area (lowered for testing)
BRIGHTNESS_RANGE = (40, 235)  # acceptable mean grayscale (wider for testing)
OUTPUT_SIZE = (224, 224)  # aligned face size (H, W)

# FaceMesh indices for outer eye corners (approx.)
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# Lazily-initialized singleton FaceMesh instance (avoids per-frame re-creation).
_FACE_MESH = None


def get_face_mesh():
    """
    Returns a persistent MediaPipe FaceMesh instance.
    Keeps a single instance to reduce overhead and silence Pylance dynamic attribute warnings.
    """
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


# Ensure FaceMesh is closed on exit to release resources
@atexit.register
def _close_face_mesh():
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


def variance_of_laplacian(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def brightness_mean(img_gray: np.ndarray) -> float:
    return float(np.mean(img_gray))


def eye_aligned_face(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Rotate + crop so that the eye-line is horizontal, then resize."""
    x, y, w, h = bbox_xywh
    # Compute eye centers from FaceMesh landmarks (pixel coords)
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]

    # Angle to rotate so the eyes are horizontal
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the whole frame around the bbox center (robust crop afterward)
    center = (x + w // 2, y + h // 2)
    M = cv2.getRotationMatrix2D(
        center, -angle, 1.0
    )  # Rotate by the negative angle so the eye-line is leveled.
    rotated = cv2.warpAffine(
        frame_bgr,
        M,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Expand bbox a bit to capture chin/forehead after rotation
    pad = int(0.2 * max(w, h))
    rx, ry = max(0, x - pad), max(0, y - pad)
    rw = min(rotated.shape[1] - rx, w + 2 * pad)
    rh = min(rotated.shape[0] - ry, h + 2 * pad)
    crop = rotated[ry : ry + rh, rx : rx + rw]

    # Final resize
    face_aligned = cv2.resize(
        crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
    )
    return face_aligned


def detect_face_and_landmarks(
    frame_bgr: np.ndarray,
) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Returns (bbox_xywh, landmarks_xy) or None if not found.
    Uses MediaPipe FaceMesh (single face).
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Use a persistent FaceMesh instance for performance
    fm = get_face_mesh()
    res = cast(Any, fm.process(rgb))

    if not getattr(res, "multi_face_landmarks", None):
        return None

    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    landmarks_xy = np.stack([xs, ys], axis=1)  # (468, 2)

    # Bounding box from landmarks (tighter than a detector bbox)
    x_min, y_min = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(ys.min(), 0, h - 1))
    x_max, y_max = int(np.clip(xs.max(), 0, w - 1)), int(np.clip(ys.max(), 0, h - 1))
    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    return bbox, landmarks_xy


def _unsharp_mask(gray: np.ndarray, amount: float = 1.5, radius: int = 1) -> np.ndarray:
    # Simple unsharp mask: sharpen edges to boost Laplacian variance on soft images
    blurred = cv2.GaussianBlur(gray, (radius * 2 + 1, radius * 2 + 1), 0)
    sharp = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _clahe(
    gray: np.ndarray, clip: float = 2.0, tiles: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    # Local contrast normalization helps low-light laptop cameras
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return clahe.apply(gray)


def assess_quality(
    frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]
) -> FaceQuality:
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox_xywh
    face = frame_bgr[y : y + bh, x : x + bw]
    gray0 = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Preprocess to help low-res/low-light webcams
    gray_eq = _clahe(gray0)
    gray = _unsharp_mask(gray_eq, amount=1.5, radius=1)

    blur = variance_of_laplacian(gray)
    bright = brightness_mean(gray0)
    bbox_ratio = (bw * bh) / float(w * h)

    reasons_list = []
    if blur < MIN_BLUR_VAR:
        reasons_list.append(f"blur {blur:.0f} < {MIN_BLUR_VAR}")
    if bright < BRIGHTNESS_RANGE[0] or bright > BRIGHTNESS_RANGE[1]:
        reasons_list.append(
            f"brightness {bright:.0f} ∉ [{BRIGHTNESS_RANGE[0]}, {BRIGHTNESS_RANGE[1]}]"
        )
    if bbox_ratio < MIN_BBOX_RATIO:
        reasons_list.append(f"bbox {bbox_ratio:.3f} < {MIN_BBOX_RATIO}")
    passed = len(reasons_list) == 0
    return FaceQuality(blur, bright, bbox_ratio, passed, "; ".join(reasons_list))


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
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=False, index=False)


def process_frame(
    frame_bgr: np.ndarray,
    user_id: str,
    out_dir: str,
    base_name: str,
    return_metrics: bool = False,
) -> Any:
    """
    Detects, quality-checks, aligns, saves face and logs metadata.
    Returns saved filepath or None if rejected/not found.
    """
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
    fname = f"{base_name}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

    meta_csv = os.path.join(os.path.dirname(out_dir), "metadata.csv")
    h, w = frame_bgr.shape[:2]
    append_metadata(
        meta_csv,
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
