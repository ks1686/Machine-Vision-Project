# make_average_face.py (simplified)
# Builds a single averaged face image from your aligned crops using robust defaults:
#  - Uses top-K sharpest straight-ahead frames (K=24, step=1)
#  - Aligns all frames to a reference face via FaceMesh landmarks
#  - Drops outliers via reprojection MAD and a rotation limit (15°)
#  - Chooses between sigma-clipped MEAN vs MEDIAN automatically using Laplacian sharpness
#  - Writes one file: <user_id>_avgface.png

from __future__ import annotations

import os
import re
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from face_processing import facemesh_vector_from_aligned

# -------------------------- Helpers --------------------------


def _variance_of_laplacian(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _lm2d_from_img(img: np.ndarray) -> Optional[np.ndarray]:
    vec = facemesh_vector_from_aligned(img)
    if vec is None:
        return None
    total = int(vec.size)
    if total % 3 != 0:
        return None
    N = total // 3
    pts = vec.reshape(N, 3).astype(np.float32)
    return pts[:, :2].copy()


def _align_to_reference(
    img: np.ndarray, lm: np.ndarray, ref_lm: np.ndarray, out_size: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if lm is None or ref_lm is None:
        return None, None
    M, _ = cv2.estimateAffinePartial2D(lm, ref_lm, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(
            lm, ref_lm, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
    if M is None:
        return None, None
    W, H = out_size[1], out_size[0]
    warped = cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return warped, M


def _step_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"_s(\d{2})_", os.path.basename(fname))
    return int(m.group(1)) if m else None


def _collect_candidates(
    user_id: str, base_dir: str, step: Optional[int]
) -> pd.DataFrame:
    meta_path = os.path.join(base_dir, "metadata.csv")
    user_dir = os.path.join(base_dir, user_id)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}")
    if not os.path.isdir(user_dir):
        raise FileNotFoundError(f"user dir not found: {user_dir}")

    df = pd.read_csv(meta_path)
    df = df[(df["user_id"] == user_id) & df["file"].notna()].copy()
    df["abs_path"] = df["file"].apply(lambda p: os.path.join(base_dir, p))
    df = df[df["abs_path"].apply(os.path.exists)].copy()
    df["step"] = df["abs_path"].apply(_step_from_filename)
    if step is not None:
        df = df[df["step"] == step]
    df = (
        df[pd.notnull(df["blur_var"])]
        .sort_values("blur_var", ascending=False)
        .reset_index(drop=True)
    )
    return df[["abs_path", "blur_var", "step"]].rename(columns={"abs_path": "file"})


def _load_images(paths: List[str]) -> np.ndarray:
    imgs: List[np.ndarray] = []
    H = W = None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if H is None:
            H, W = img.shape[:2]
        elif img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError("No images could be loaded to average.")
    return np.stack(imgs, axis=0)


def _unsharp(x: np.ndarray, amount: float = 0.6, radius: int = 1) -> np.ndarray:
    x8 = np.clip(x, 0, 255).astype(np.uint8)
    k = radius * 2 + 1
    blur = cv2.GaussianBlur(x8, (k, k), 0)
    y = cv2.addWeighted(x8, 1 + amount, blur, -amount, 0).astype(np.float32)
    return y


# -------------------------- Main --------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Create a single averaged face for a user (median or mean chosen automatically)."
    )
    parser.add_argument("user_id")
    parser.add_argument("--base_dir", default="registered_faces")
    args = parser.parse_args()

    # Defaults requested
    TOP_K = 24
    STEP = 1  # straight ahead
    ROT_DEG_MAX = 15.0  # in-plane rotation limit
    KEEP_ADAPTIVE = True  # MAD-based outlier drop

    # 1) Pick candidates
    step = None if STEP == 0 else STEP
    df = _collect_candidates(args.user_id, args.base_dir, step)
    if len(df) == 0:
        raise RuntimeError("No candidate aligned images found.")
    use = df.iloc[: min(TOP_K, len(df))].copy()

    raw_imgs = _load_images(use["file"].tolist())
    H, W = raw_imgs.shape[1:3]

    # 2) Reference (sharpest)
    ref_path = use["file"].iloc[0]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_img is None or ref_img.shape[:2] != (H, W):
        ref_img = raw_imgs[0].astype(np.uint8)
        if ref_img.shape[:2] != (H, W):
            ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    ref_lm = _lm2d_from_img(ref_img)

    # 3) Align all candidates, compute reprojection error, drop outliers
    aligned: List[Tuple[np.ndarray, float]] = []
    for p in use["file"].tolist():
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        lm = _lm2d_from_img(img)
        warped, M = (
            _align_to_reference(img, lm, ref_lm, (H, W))
            if (lm is not None and ref_lm is not None)
            else (None, None)
        )

        # rotation gate
        if M is not None:
            a, b = float(M[0, 0]), float(M[0, 1])
            rot = float(np.degrees(np.arctan2(b, a)))
            if abs(rot) > ROT_DEG_MAX:
                continue

        # reprojection error
        if M is not None and lm is not None and ref_lm is not None:
            proj = (lm @ M[:, :2].T) + M[:, 2]
            err = float(np.linalg.norm(proj - ref_lm, axis=1).mean())
        else:
            err = float("inf")

        aligned.append(((warped if warped is not None else img), err))

    if not aligned:
        raise RuntimeError("No frames survived alignment filtering.")

    # adaptive keep via MAD
    if KEEP_ADAPTIVE:
        errs = np.array([e for _, e in aligned], dtype=np.float32)
        finite = np.isfinite(errs)
        items = [aligned[i] for i, ok in enumerate(finite) if ok]
        errs = errs[finite]
        med = float(np.median(errs))
        mad = float(np.median(np.abs(errs - med)) + 1e-6)
        thr = med + 2.5 * (1.4826 * mad)
        kept = [it for it in items if it[1] <= thr]
        if len(kept) < max(6, int(0.5 * len(items))):
            k = max(6, int(0.9 * len(items)))
            kept = sorted(items, key=lambda t: t[1])[:k]
        aligned = kept

    imgs = np.stack([a[0].astype(np.float32) for a in aligned], axis=0)
    K = imgs.shape[0]

    # 4) Build face mask (weights only) from reference landmarks
    mask3 = None
    if ref_lm is not None:
        hull = cv2.convexHull(ref_lm.astype(np.int32))
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        mask3 = np.dstack([mask, mask, mask]).astype(np.float32)

    # 5) Compute two candidates: sigma-clipped mean vs median
    med_img = np.median(imgs, axis=0)
    mad_img = np.median(np.abs(imgs - med_img), axis=0) + 1e-6
    z = np.abs(imgs - med_img) / (1.4826 * mad_img)
    w_clip = (z < 2.5).astype(np.float32)
    w = w_clip * (mask3[None, ...] if mask3 is not None else 1.0)
    mean_img = (imgs * w).sum(axis=0) / (w.sum(axis=0) + 1e-6)

    # 6) Choose sharper composite using Laplacian variance inside the face mask
    def _score(img_bgr_f32: np.ndarray) -> float:
        g = cv2.cvtColor(
            np.clip(img_bgr_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
        )
        if mask3 is not None:
            g = (g.astype(np.float32) * mask3[..., 0]).astype(np.uint8)
        return _variance_of_laplacian(g)

    mean_score = _score(mean_img)
    median_score = _score(med_img)
    chosen = mean_img if mean_score >= median_score else med_img

    # Light unsharp to counteract averaging blur
    out_img = _unsharp(chosen, amount=0.5, radius=1)

    # 7) Write one output
    out_dir = os.path.join(args.base_dir, args.user_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.user_id}_avgface.png")
    cv2.imwrite(
        out_path,
        np.clip(out_img, 0, 255).astype(np.uint8),
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )

    method = "mean" if mean_score >= median_score else "median"
    print("[average] wrote:")
    print(f"  out:    {out_path}  (method={method}, K={K}, HxW={H}x{W})")


if __name__ == "__main__":
    main()
