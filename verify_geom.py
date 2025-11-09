# verify_geom.py
import os
import json
import numpy as np
from geom_embed import (
    geometry_embedding_from_aligned,
)  # optional if you need to embed probes from images
import cv2  # only needed if using CLI with image path


def _mahalanobis(x, mu, cov_inv, diag_inv_fallback=None):
    d = x - mu
    try:
        m2 = float(d @ cov_inv @ d)
        if np.isfinite(m2):
            return m2
    except Exception:
        pass
    # Fallback: diagonal Mahalanobis or Euclidean
    if diag_inv_fallback is None:
        return float(np.dot(d, d))
    return float(np.sum((d * d) * diag_inv_fallback))


# Back-compat wrapper used by older imports (not recommended for new code)
def _mahalanobis_legacy(x, mu, cov_inv):
    return _mahalanobis(x, mu, cov_inv, diag_inv_fallback=None)


def load_geom_model(user_id: str, model_dir="models"):
    path = os.path.join(model_dir, f"{user_id}_geom.json")
    with open(path, "r") as f:
        M = json.load(f)
    mu = np.array(M["mu"], np.float32)
    cov = np.array(M["cov"], np.float32)
    # Prefer pseudo-inverse for stability on high-dim, low-N regimes
    cov_inv = np.linalg.pinv(cov, rcond=1e-6).astype(np.float32)
    # Precompute diagonal inverse as a fallback (variance floor)
    var = np.clip(np.diag(cov), 1e-4, None).astype(np.float32)
    var_inv = (1.0 / var).astype(np.float32)
    return mu, cov_inv, {"cov_diag_inv": var_inv, "meta": M}


def load_threshold(user_id: str, model_dir="models", fallback=0.0):
    p = os.path.join(model_dir, f"{user_id}_geom_thresh.json")
    if not os.path.exists(p):
        return fallback
    with open(p, "r") as f:
        return float(json.load(f)["geom_thresh"])


def verify_geom(user_id: str, probe_aligned_png: str, model_dir="models"):
    mu, cov_inv, extras = load_geom_model(user_id, model_dir)
    var_inv = extras["cov_diag_inv"]
    thr = load_threshold(user_id, model_dir)

    img = cv2.imread(probe_aligned_png)
    if img is None:
        return {"ok": False, "reason": f"cannot read {probe_aligned_png}"}
    e = geometry_embedding_from_aligned(img)
    if e is None:
        return {"ok": False, "reason": "no face in probe"}

    s = -_mahalanobis(e, mu, cov_inv, diag_inv_fallback=var_inv)
    return {"ok": bool(s >= thr), "s_geom": float(s), "thresh": float(thr)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    ap.add_argument("probe_aligned_png")
    args = ap.parse_args()
    out = verify_geom(args.user_id, args.probe_aligned_png)
    print(out)
