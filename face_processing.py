"""
face_processing.py
------------------
Utilities for face detection, quality assessment, alignment, and saving.

Editable parameters:
- MIN_BLUR_VAR: laptop webcams often yield 5–20; raise for stricter blur checks.
- BRIGHTNESS_RANGE: widen for dim rooms, tighten for controlled lighting.
- MIN_BBOX_RATIO: face area / frame area; lower if users sit far from camera.
- OUTPUT_SIZE: final aligned crop size (H, W).
# NOTE: We also save the original raw frame and full warp metadata per accepted capture for 3D reconstruction.
"""

from __future__ import annotations

import atexit
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# -------- Quality / alignment parameters (tune here as needed) --------
MIN_BLUR_VAR = 10.0  # variance of Laplacian on sharpened ROI
MIN_BBOX_RATIO = 0.04  # min face area / frame area
BRIGHTNESS_RANGE = (40, 235)
OUTPUT_SIZE = (512, 512)  # (H, W)  # higher-res aligned crops for better texture

# MediaPipe indices for outer eye corners.
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# Persistent FaceMesh instances: video tracking vs one-shot stills.
_FACE_MESH_VIDEO = None
_FACE_MESH_STILL = None
_CLAHE = None


def _make_face_mesh(*, static_image_mode: bool):
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def get_face_mesh(*, static_image_mode: bool = False):
    """Return a persistent MediaPipe FaceMesh instance (single-face)."""
    global _FACE_MESH_VIDEO, _FACE_MESH_STILL
    if static_image_mode:
        if _FACE_MESH_STILL is None:
            _FACE_MESH_STILL = _make_face_mesh(static_image_mode=True)
        return _FACE_MESH_STILL
    if _FACE_MESH_VIDEO is None:
        _FACE_MESH_VIDEO = _make_face_mesh(static_image_mode=False)
    return _FACE_MESH_VIDEO


def get_clahe(clip: float = 2.0, tiles: tuple[int, int] = (8, 8)):
    """Reuse one CLAHE instance for quality checks and still-image detection."""
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return _CLAHE


@atexit.register
def _close_face_mesh():
    """Close FaceMesh on interpreter exit to release resources."""
    global _FACE_MESH_VIDEO, _FACE_MESH_STILL
    for mesh in (_FACE_MESH_VIDEO, _FACE_MESH_STILL):
        if mesh is not None:
            try:
                mesh.close()
            except Exception:
                pass
    _FACE_MESH_VIDEO = None
    _FACE_MESH_STILL = None


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
    gray: np.ndarray, clip: float = 2.0, tiles: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Local contrast normalization helps low-light laptop cameras."""
    return get_clahe(clip=clip, tiles=tiles).apply(gray)


# ---- Core ops ----


def detect_face_and_landmarks(
    frame_bgr: np.ndarray,
    *,
    static_image_mode: bool = False,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Return (bbox_xywh, landmarks_xy) in pixel coordinates; None if not found."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fm = get_face_mesh(static_image_mode=static_image_mode)
    res = cast(Any, fm.process(rgb))
    if not getattr(res, "multi_face_landmarks", None):
        return None

    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    landmarks_xy = np.stack([xs, ys], axis=1)  # (478, 2) with refine_landmarks

    x_min, y_min = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(ys.min(), 0, h - 1))
    x_max, y_max = int(np.clip(xs.max(), 0, w - 1)), int(np.clip(ys.max(), 0, h - 1))
    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    return bbox, landmarks_xy


def rotated_crop_roi(
    bbox_xywh: tuple[int, int, int, int],
    angle_deg: float,
    frame_wh: tuple[int, int],
    pad_ratio: float = 0.3,
) -> tuple[int, int, int, int]:
    """Axis-aligned crop of a bbox after rotation about its center."""
    x, y, w, h = bbox_xywh
    frame_w, frame_h = frame_wh
    cx = x + w / 2.0
    cy = y + h / 2.0
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
    )
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    ones = np.ones((4, 1), dtype=np.float32)
    rotated = (M @ np.hstack([corners, ones]).T).T
    rx0 = int(np.floor(rotated[:, 0].min()))
    ry0 = int(np.floor(rotated[:, 1].min()))
    rx1 = int(np.ceil(rotated[:, 0].max()))
    ry1 = int(np.ceil(rotated[:, 1].max()))
    pad = int(pad_ratio * max(w, h))
    rx = max(0, rx0 - pad)
    ry = max(0, ry0 - pad)
    rw = min(frame_w - rx, (rx1 - rx0) + 2 * pad)
    rh = min(frame_h - ry, (ry1 - ry0) + 2 * pad)
    return int(rx), int(ry), int(max(1, rw)), int(max(1, rh))


def eye_aligned_face(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    target_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate so the eye line is horizontal, crop a padded ROI, then resize.
    Returns (aligned_bgr, meta) where meta contains the full warp and ROI info.
    """
    x, y, w, h = bbox_xywh
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    center = (float(x + w / 2.0), float(y + h / 2.0))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        frame_bgr,
        M,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    rx, ry, rw, rh = rotated_crop_roi(
        bbox_xywh,
        angle_deg=angle,
        frame_wh=(rotated.shape[1], rotated.shape[0]),
        pad_ratio=0.3,
    )
    crop = rotated[ry : ry + rh, rx : rx + rw]
    pad = int(0.3 * max(w, h))

    aligned = cv2.resize(
        crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4
    )

    meta: dict[str, Any] = {
        "angle_deg": angle,
        "M": [
            float(M[0, 0]),
            float(M[0, 1]),
            float(M[0, 2]),
            float(M[1, 0]),
            float(M[1, 1]),
            float(M[1, 2]),
        ],
        "roi_x": int(rx),
        "roi_y": int(ry),
        "roi_w": int(rw),
        "roi_h": int(rh),
        "pad_px": int(pad),
        "target_h": int(target_size[0]),
        "target_w": int(target_size[1]),
        "left_eye_x": float(left_eye[0]),
        "left_eye_y": float(left_eye[1]),
        "right_eye_x": float(right_eye[0]),
        "right_eye_y": float(right_eye[1]),
    }
    return aligned, meta


def assess_quality(
    frame_bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int]
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
                "file_raw",
                "blur_var",
                "brightness",
                "bbox_ratio",
                "frame_w",
                "frame_h",
                "aligned_h",
                "aligned_w",
                "angle_deg",
                "M00",
                "M01",
                "M02",
                "M10",
                "M11",
                "M12",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h",
                "pad_px",
                "left_eye_x",
                "left_eye_y",
                "right_eye_x",
                "right_eye_y",
            ]
        ).to_csv(path, index=False)


def append_metadata(csv_path: str, row: dict):
    ensure_csv(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)


# ---- Public API ----


def resolve_user_id(
    argv: Sequence[str],
    prompt: Callable[[str], str] = input,
) -> str:
    """Prefer CLI user id; otherwise prompt."""
    if len(argv) >= 2 and argv[1].strip():
        return argv[1].strip()
    return prompt("Enter User ID: ").strip()


def align_face(
    frame_bgr: np.ndarray,
    *,
    static_image_mode: bool = False,
) -> tuple[np.ndarray | None, FaceQuality, dict[str, Any] | None]:
    """Detect, quality-check, and align without touching disk."""
    detection = detect_face_and_landmarks(
        frame_bgr, static_image_mode=static_image_mode
    )
    if detection is None:
        return None, FaceQuality(0.0, 0.0, 0.0, False, "no face detected"), None

    bbox, landmarks = detection
    quality = assess_quality(frame_bgr, bbox)
    if not quality.passed:
        return None, quality, None

    aligned, meta = eye_aligned_face(frame_bgr, landmarks, bbox, OUTPUT_SIZE)
    return aligned, quality, meta


def process_frame(
    frame_bgr: np.ndarray,
    user_id: str,
    out_dir: str,
    base_name: str,
    return_metrics: bool = False,
    write_metadata: bool = True,
    write_raw: bool = True,
    aligned: np.ndarray | None = None,
    quality: FaceQuality | None = None,
    meta: dict[str, Any] | None = None,
) -> Any:
    """Detect, quality-check, align, and save one face frame. Optionally return metrics."""
    if aligned is None or quality is None or meta is None:
        aligned, quality, meta = align_face(frame_bgr, static_image_mode=False)
    if aligned is None or quality is None or meta is None:
        if return_metrics:
            return None, quality
        return None

    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{base_name}.png")
    fraw = os.path.join(out_dir, "raw", f"{base_name}_raw.png")
    cv2.imwrite(fpath, aligned, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if write_raw:
        os.makedirs(os.path.dirname(fraw), exist_ok=True)
        cv2.imwrite(fraw, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    if write_metadata:
        h, w = frame_bgr.shape[:2]
        append_metadata(
            os.path.join(os.path.dirname(out_dir), "metadata.csv"),
            {
                "timestamp": int(time.time()),
                "user_id": user_id,
                "file": os.path.relpath(fpath, os.path.dirname(out_dir)),
                "file_raw": os.path.relpath(fraw, os.path.dirname(out_dir)),
                "blur_var": quality.blur_var,
                "brightness": quality.brightness,
                "bbox_ratio": quality.bbox_ratio,
                "frame_w": w,
                "frame_h": h,
                "aligned_h": OUTPUT_SIZE[0],
                "aligned_w": OUTPUT_SIZE[1],
                "angle_deg": meta["angle_deg"],
                "M00": meta["M"][0],
                "M01": meta["M"][1],
                "M02": meta["M"][2],
                "M10": meta["M"][3],
                "M11": meta["M"][4],
                "M12": meta["M"][5],
                "roi_x": meta["roi_x"],
                "roi_y": meta["roi_y"],
                "roi_w": meta["roi_w"],
                "roi_h": meta["roi_h"],
                "pad_px": meta["pad_px"],
                "left_eye_x": meta["left_eye_x"],
                "left_eye_y": meta["left_eye_y"],
                "right_eye_x": meta["right_eye_x"],
                "right_eye_y": meta["right_eye_y"],
            },
        )

    if return_metrics:
        return fpath, quality

    return fpath


# ---------------------------------------------------------------------------
# Public helper: facemesh_vector_from_aligned()
# ---------------------------------------------------------------------------


def facemesh_vector_from_aligned(img):
    """Return a flattened landmark vector from an aligned 224/512/768 face crop."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = get_clahe().apply(gray)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for try_img in [img, color]:
        rgb = cv2.cvtColor(try_img, cv2.COLOR_BGR2RGB)
        res = get_face_mesh(static_image_mode=True).process(rgb)
        if getattr(res, "multi_face_landmarks", None):
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h, p.z] for p in lm], dtype=np.float32)
            return pts.flatten()

    return None
