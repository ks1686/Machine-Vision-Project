# enroll_geom.py
import os
import glob
import json
import numpy as np
import cv2
from geom_embed import geometry_embedding_from_aligned

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _load_aligned_images(user_id: str, base_dir="registered_faces", max_imgs=None):
    user_dir = os.path.join(base_dir, user_id)
    paths = sorted(glob.glob(os.path.join(user_dir, "*.png")))
    if max_imgs:
        paths = paths[:max_imgs]
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append((p, img))
    return imgs


def _fit_stats(X: np.ndarray, eps: float = 1e-6):
    # Mean
    mu = X.mean(axis=0).astype(np.float32)
    # Sample covariance (features in columns)
    cov = np.cov(X, rowvar=False).astype(np.float32)
    # Ridge shrinkage: alpha * I, where alpha is 1% of average variance
    avg_var = float(np.trace(cov) / cov.shape[0]) if cov.size else 0.0
    alpha = max(eps, 0.01 * avg_var)
    cov += alpha * np.eye(cov.shape[0], dtype=np.float32)
    return mu, cov


def enroll_geom(user_id: str, base_dir="registered_faces", max_imgs=None):
    pairs = _load_aligned_images(user_id, base_dir, max_imgs)
    if not pairs:
        raise RuntimeError(f"No aligned PNGs found for user '{user_id}' in {base_dir}/")

    embs = []
    for _, img in pairs:
        e = geometry_embedding_from_aligned(img)
        if e is not None:
            embs.append(e)
    if len(embs) < 5:
        raise RuntimeError(f"Too few usable samples ({len(embs)}). Capture more.")

    X = np.stack(embs, axis=0)  # (N, D)
    mu, cov = _fit_stats(X)

    model = {
        "user_id": user_id,
        "type": "geom_v1",
        "dim": int(mu.shape[0]),
        "mu": mu.tolist(),
        "cov": cov.tolist(),
        "notes": "Mahalanobis distance with pseudo-inverse; diagonal fallback available.",
    }
    out_path = os.path.join(MODELS_DIR, f"{user_id}_geom.json")
    with open(out_path, "w") as f:
        json.dump(model, f)
    print(f"[enroll] wrote {out_path} (N={len(embs)} samples)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    ap.add_argument("--max_imgs", type=int, default=None)
    args = ap.parse_args()
    enroll_geom(args.user_id, max_imgs=args.max_imgs)
