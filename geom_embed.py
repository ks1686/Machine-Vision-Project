# geom_embed.py
import numpy as np
from face_processing import (
    facemesh_vector_from_aligned,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
)

# Kept for reference; do not rely on this for reshaping:
LMN = 468


def _normalize_xyz(pts: np.ndarray) -> np.ndarray:
    """
    pts: (N,3) in crop coords.
    Returns pose/translation/scale normalized array (N,3).
    """
    xy = pts[:, :2].astype(np.float32).copy()
    z = pts[:, 2:3].astype(np.float32).copy()

    # center by centroid (2D)
    xy -= xy.mean(axis=0, keepdims=True)

    # scale by inter-ocular distance (robust to small pose changes)
    le = pts[LEFT_EYE_IDX, :2]
    re = pts[RIGHT_EYE_IDX, :2]
    iodd = float(np.linalg.norm(re - le) + 1e-6)
    xy /= iodd

    # z: standardize (illumination/back-projection robustness)
    z = (z - z.mean()) / (z.std() + 1e-6)

    return np.concatenate([xy, z], axis=1)


def geometry_embedding_from_aligned(aligned_bgr: np.ndarray) -> np.ndarray | None:
    """
    aligned_bgr: aligned crop read via cv2.imread(...).
    returns: (N*3,) float32 where N is the number of landmarks returned by MediaPipe
             (468 without iris; 478 with iris when refine_landmarks=True). Returns None if no face.
    """
    vec = facemesh_vector_from_aligned(aligned_bgr)
    if vec is None:
        return None

    # MediaPipe FaceMesh returns 468 landmarks; with refine_landmarks=True it returns 478 (adds iris).
    total = int(vec.size)
    if total % 3 != 0:
        # Unexpected shape; bail out safely
        return None
    N = total // 3

    pts = vec.reshape(N, 3).astype(np.float32)

    # Sanity check: required eye indices must exist for normalization
    if LEFT_EYE_IDX >= N or RIGHT_EYE_IDX >= N:
        return None

    ptsn = _normalize_xyz(pts)
    return ptsn.reshape(-1).astype(np.float32)
