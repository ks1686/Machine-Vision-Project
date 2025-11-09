# mesh_textured.py
# Build a textured face mesh (.obj + .mtl) from your averaged aligned face image.
# Usage:
#   python mesh_textured.py ks1686
# Optional:
#   python mesh_textured.py ks1686 --base_dir registered_faces --models_dir models --tex_path registered_faces/ks1686/ks1686_avgface.png

from __future__ import annotations
import os
import argparse
import shutil
import numpy as np
import cv2

from scipy.spatial import Delaunay
from face_processing import facemesh_vector_from_aligned


def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def _load_texture(
    user_id: str, base_dir: str, tex_path: str | None
) -> tuple[np.ndarray, str]:
    # Default texture is the averaged face produced earlier
    if tex_path is None:
        tex_path = os.path.join(base_dir, user_id, f"{user_id}_avgface.png")
    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"Texture image not found: {tex_path}")
    img = cv2.imread(tex_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read texture at {tex_path}")
    return img, tex_path


def _landmarks_from_image(img_bgr: np.ndarray) -> np.ndarray:
    vec = facemesh_vector_from_aligned(img_bgr)
    if vec is None or vec.size % 3 != 0:
        raise RuntimeError("FaceMesh landmarks not found in texture image.")
    pts = vec.reshape(-1, 3).astype(
        np.float32
    )  # (N,3) with x,y in pixels; z in relative units
    return pts


def _normalize_vertices_for_view(pts_xyz: np.ndarray) -> np.ndarray:
    """Center to origin; scale to unit-ish size for nicer viewing (preserve aspect)."""
    V = pts_xyz.astype(np.float32).copy()
    # Center XY around 0 (leave Z relative)
    V[:, 0:2] -= V[:, 0:2].mean(axis=0, keepdims=True)
    # Normalize XY scale by 95th percentile radius
    r = np.sqrt((V[:, 0] ** 2 + V[:, 1] ** 2))
    s = np.percentile(r, 95) + 1e-6
    V[:, 0:2] /= s
    # Z: standardize a bit so the mesh isn’t paper-thin
    z = V[:, 2:3]
    z = (z - z.mean()) / (z.std() + 1e-6)
    V[:, 2:3] = 0.8 * z
    V[:, 1] *= -1  # flip vertical orientation for upright view
    return V


def _delaunay_faces(xy: np.ndarray) -> np.ndarray:
    tri = Delaunay(xy)
    return tri.simplices.astype(np.int32)  # (M,3)


def _write_obj_mtl(
    out_obj: str,
    out_mtl: str,
    tex_rel: str,
    V: np.ndarray,
    UV: np.ndarray,
    F: np.ndarray,
):
    """
    OBJ with v/vt; faces reference both (no normals needed).
    UV coords must be in [0,1]; note OBJ uses v downwards positive, same as image if we invert.
    """
    with open(out_mtl, "w") as m:
        m.write("newmtl face_tex\n")
        m.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
        m.write(f"map_Kd {tex_rel}\n")

    with open(out_obj, "w") as f:
        f.write(f"mtllib {os.path.basename(out_mtl)}\n")
        f.write("usemtl face_tex\n")

        # Vertices
        for x, y, z in V:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        # Texture coordinates
        for u, v in UV:
            f.write(f"vt {u:.6f} {v:.6f}\n")

        # Faces (OBJ is 1-indexed; use v/vt)
        for i, j, k in F:
            i1, j1, k1 = i + 1, j + 1, k + 1
            f.write(f"f {i1}/{i1} {j1}/{j1} {k1}/{k1}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Create a textured face mesh (OBJ+MTL) from the averaged face image."
    )
    ap.add_argument("user_id")
    ap.add_argument(
        "--base_dir",
        default="registered_faces",
        help="Where averaged texture lives (user folder).",
    )
    ap.add_argument(
        "--models_dir", default="models", help="Where to write OBJ/MTL/texture copy."
    )
    ap.add_argument(
        "--tex_path",
        default=None,
        help="Explicit texture path (defaults to <base_dir>/<user>/<user>_avgface.png).",
    )
    args = ap.parse_args()

    _ensure_dirs(args.models_dir)

    # 1) Load texture & landmarks
    tex_img, tex_path = _load_texture(args.user_id, args.base_dir, args.tex_path)
    H, W = tex_img.shape[:2]
    lm = _landmarks_from_image(tex_img)  # (N,3) in pixel coords
    xy = lm[:, :2].copy()
    z = lm[:, 2:3].copy()

    # 2) Build UVs from pixel coords (normalize to [0,1], flip v)
    UV = np.empty((xy.shape[0], 2), dtype=np.float32)
    UV[:, 0] = xy[:, 0] / float(W)  # u = x / W
    UV[:, 1] = 1.0 - (xy[:, 1] / float(H))  # v flipped for OBJ

    # 3) Triangles via 2D Delaunay on (x,y) pixel coords (robust for surface)
    F = _delaunay_faces(xy)  # (M,3)

    # 4) 3D vertices for viewing (center/scale)
    V = _normalize_vertices_for_view(np.concatenate([xy, z], axis=1))  # (N,3)

    # 5) Write OBJ + MTL + copy texture
    user = args.user_id
    obj_path = os.path.join(args.models_dir, f"{user}_face.obj")
    mtl_path = os.path.join(args.models_dir, f"{user}_face.mtl")
    tex_copy = os.path.join(args.models_dir, f"{user}_face_texture.png")

    # Make a copy of the texture next to the OBJ (keeps relative path simple)
    if os.path.abspath(tex_copy) != os.path.abspath(tex_path):
        shutil.copyfile(tex_path, tex_copy)

    _write_obj_mtl(obj_path, mtl_path, os.path.basename(tex_copy), V, UV, F)

    print("[mesh_textured] wrote:")
    print(f"  OBJ: {obj_path}")
    print(f"  MTL: {mtl_path}")
    print(f"  TEX: {tex_copy}")
    print(f"  verts={V.shape[0]}, faces={F.shape[0]}")


if __name__ == "__main__":
    main()
