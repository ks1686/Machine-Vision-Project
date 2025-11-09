# calibrate_geom.py
import os
import glob
import json
import random
import numpy as np
import cv2
from geom_embed import geometry_embedding_from_aligned
from verify_geom import load_geom_model, _mahalanobis


def _collect_embs(user_id: str, base_dir="registered_faces", max_imgs=80):
    user_dir = os.path.join(base_dir, user_id)
    paths = sorted(glob.glob(os.path.join(user_dir, "*.png")))
    random.shuffle(paths)
    paths = paths[:max_imgs]
    E = []
    for p in paths:
        img = cv2.imread(p)
        e = geometry_embedding_from_aligned(img)
        if e is not None:
            E.append(e)
    return np.array(E, dtype=np.float32)


def calibrate(user_id: str, model_dir="models", base_dir="registered_faces"):
    mu, cov_inv, extras = load_geom_model(user_id, model_dir)
    var_inv = extras["cov_diag_inv"]
    E = _collect_embs(user_id, base_dir)
    if len(E) < 10:
        raise RuntimeError(f"Need >=10 aligned samples; found {len(E)}")

    scores = np.array(
        [-(_mahalanobis(e, mu, cov_inv, diag_inv_fallback=var_inv)) for e in E],
        np.float32,
    )
    thr = float(np.percentile(scores, 10.0))  # conservative; tune later

    out = dict(
        user_id=user_id,
        geom_thresh=thr,
        genuine_mean=float(scores.mean()),
        genuine_std=float(scores.std()),
    )
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{user_id}_geom_thresh.json"), "w") as f:
        json.dump(out, f)
    print(f"[calibrate] thr={thr:.3f} from N={len(scores)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    args = ap.parse_args()
    calibrate(args.user_id)
