# make_average_face.py
import os
import re
import argparse
import numpy as np
import cv2
import pandas as pd
from typing import List
from face_processing import facemesh_vector_from_aligned


def _lm2d_from_img(img: np.ndarray) -> np.ndarray | None:
    """Return (N,2) float32 facial landmarks in the image's pixel coords using MediaPipe FaceMesh.
    Works for N=468 or N=478 (refine_landmarks=True). Returns None if not detected."""
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
    img: np.ndarray, lm: np.ndarray, ref_lm: np.ndarray, out_size: tuple[int, int]
) -> np.ndarray | None:
    """Estimate a similarity transform that maps lm->ref_lm and warp `img` to `out_size`.
    Uses cv2.estimateAffinePartial2D on all available correspondences for robustness."""
    # Guard
    if lm is None or ref_lm is None:
        return None
    # Estimate 2x3 similarity transform (rotation+scale+translation)
    M, inliers = cv2.estimateAffinePartial2D(lm, ref_lm, method=cv2.LMEDS)
    if M is None:
        # fallback to RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            lm, ref_lm, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
    if M is None:
        return None
    W, H = out_size[1], out_size[0]
    warped = cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return warped


def _step_from_filename(fname: str) -> int | None:
    m = re.search(r"_s(\d{2})_", os.path.basename(fname))
    return int(m.group(1)) if m else None


def _collect_candidates(user_id: str, base_dir: str, step: int | None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['file','blur_var','step']
    Uses metadata.csv to find aligned PNGs for this user.
    """
    meta_path = os.path.join(base_dir, "metadata.csv")
    user_dir = os.path.join(base_dir, user_id)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}")
    if not os.path.isdir(user_dir):
        raise FileNotFoundError(f"user dir not found: {user_dir}")

    df = pd.read_csv(meta_path)
    # Keep only this user and rows that reference aligned files (not raw)
    df = df[(df["user_id"] == user_id) & df["file"].notna()].copy()

    # Build absolute path to aligned file
    # metadata.csv stores 'file' relative to base_dir (parent of user dir)
    df["abs_path"] = df["file"].apply(lambda p: os.path.join(base_dir, p))
    df["exists"] = df["abs_path"].apply(os.path.exists)
    df = df[df["exists"]].copy()

    # Add step from filename if present
    df["step"] = df["abs_path"].apply(lambda p: _step_from_filename(p))

    # Filter by requested step if provided
    if step is not None:
        df = df[df["step"] == step]

    # Drop NaN blur_var (shouldn’t happen, but be safe)
    df = df[pd.notnull(df["blur_var"])]

    # Sort best-first by sharpness
    df = df.sort_values("blur_var", ascending=False).reset_index(drop=True)
    return df[["abs_path", "blur_var", "step"]].rename(columns={"abs_path": "file"})


def _load_images(paths: List[str]) -> np.ndarray:
    imgs = []
    H = W = None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if H is None:
            H, W = img.shape[:2]
        else:
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError("No images could be loaded to average.")
    return np.stack(imgs, axis=0)  # (N,H,W,3)


def _write(path: str, img_f32: np.ndarray):
    img_u8 = np.clip(img_f32, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def main():
    ap = argparse.ArgumentParser(
        description="Create mean/median aligned face for a user."
    )
    ap.add_argument("user_id")
    ap.add_argument(
        "--base_dir",
        default="registered_faces",
        help="Base dir that contains user subfolder and metadata.csv",
    )
    ap.add_argument("--top_k", type=int, default=24, help="Use top-K sharpest images")
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Pose step to use (1 = look straight); use 0 to allow any step",
    )
    ap.add_argument(
        "--weighted", action="store_true", help="Weight the mean by blur_var"
    )
    args = ap.parse_args()

    step = None if args.step == 0 else args.step
    df = _collect_candidates(args.user_id, args.base_dir, step)

    if len(df) == 0:
        raise RuntimeError("No candidates found; try --step 0 to use all poses.")
    use = df.iloc[: min(args.top_k, len(df))].copy()

    # Load candidate images
    raw_imgs = _load_images(use["file"].tolist())  # (N,H,W,3)
    N = raw_imgs.shape[0]

    # --- Geometric alignment to a reference face ---
    # Pick the sharpest image (first in `use`) as reference
    ref_path = use["file"].iloc[0]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    H, W = raw_imgs.shape[1:3]
    if ref_img is None or ref_img.shape[:2] != (H, W):
        ref_img = (
            cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
            if ref_img is not None
            else raw_imgs[0].astype(np.uint8)
        )
    ref_lm = _lm2d_from_img(ref_img)

    aligned_list = []
    for i, p in enumerate(use["file"].tolist()):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        lm = _lm2d_from_img(img)
        warped = (
            _align_to_reference(img, lm, ref_lm, (H, W))
            if lm is not None and ref_lm is not None
            else None
        )
        aligned_list.append(warped if warped is not None else img)

    imgs = np.stack([a.astype(np.float32) for a in aligned_list], axis=0)

    # MEAN (optionally weighted by sharpness)
    if args.weighted:
        w = use["blur_var"].to_numpy(dtype=np.float32)
        w = w[: imgs.shape[0]]
        w = w / (w.sum() + 1e-6)
        mean_img = np.tensordot(w, imgs, axes=(0, 0))
    else:
        mean_img = imgs.mean(axis=0)

    # MEDIAN (more robust to outliers)
    median_img = np.median(imgs, axis=0)

    # Mild unsharp mask to counteract averaging blur
    def unsharp(x, amount=0.6, radius=1):
        x8 = np.clip(x, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(x8, (radius * 2 + 1, radius * 2 + 1), 0)
        y = cv2.addWeighted(x8, 1 + amount, blur, -amount, 0).astype(np.float32)
        return y

    mean_img_us = unsharp(mean_img, amount=0.6, radius=1)
    median_img_us = unsharp(median_img, amount=0.4, radius=1)

    out_dir = os.path.join(args.base_dir, args.user_id)
    os.makedirs(out_dir, exist_ok=True)
    mean_path = os.path.join(out_dir, f"{args.user_id}_mean_aligned.png")
    median_path = os.path.join(out_dir, f"{args.user_id}_median_aligned.png")
    _write(mean_path, mean_img_us)
    _write(median_path, median_img_us)

    print(
        f"[average] wrote:\n  mean:   {mean_path}\n  median: {median_path}\n  used N={N} images (step={'any' if step is None else step})"
    )


if __name__ == "__main__":
    main()
