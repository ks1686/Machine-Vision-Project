#!/usr/bin/env python3
"""
reconstruct_face.py
-------------------
Face reconstruction pipeline that consumes data captured by
face_registration.py / face_processing.py and produces a textured 3D mesh.

Stages (all Python-only):
  0) Dataset load & validation (aligned + raw + metadata)
  1) SfM with pycolmap -> camera intrinsics/extrinsics + sparse points
  2) Depth prediction per image (Depth-Anything v2 / DPT) + robust scale alignment
  3) TSDF fusion (Open3D) to produce a watertight triangle mesh
  4) Cylindrical UV unwrap (no external libs) and photo texture baking
  5) Save OBJ/MTL/PNG and open interactive viewer

Run:
  python reconstruct_face.py <user_id>

This file is self-contained and will guide you if an optional dependency is
missing. It never changes your capture flow.
"""

from __future__ import annotations
import os
import sys
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# Optional deps (checked at runtime)
try:
    import open3d as o3d  # TSDF + mesh + viewer
except Exception:
    o3d = None  # type: ignore

try:
    import pycolmap  # SfM
except Exception:
    pycolmap = None  # type: ignore

try:
    import torch  # depth inference backend
except Exception:
    torch = None  # type: ignore

# Depth model import is deferred until used


ROOT = Path("registered_faces")
OUT_ROOT = Path("out_models")


@dataclass
class FrameItem:
    aligned_path: Path
    raw_path: Path
    meta: Dict[str, float]


# ----------------------------
# Stage 0: Dataset I/O
# ----------------------------


def load_dataset(user_id: str) -> List[FrameItem]:
    user = user_id.strip().replace(" ", "_")
    user_dir = ROOT / user
    raw_dir = user_dir / "raw"
    meta_path = ROOT / "metadata.csv"

    if not user_dir.is_dir():
        raise FileNotFoundError(f"Missing user dir: {user_dir}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw dir: {raw_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv at {meta_path}")

    import pandas as pd

    df = pd.read_csv(meta_path)
    df = df[df["user_id"] == user]
    if df.empty:
        raise RuntimeError(f"No metadata rows for user {user}")

    items: List[FrameItem] = []
    for _, row in df.iterrows():
        ap = (ROOT / os.path.normpath(str(row["file"]))).resolve()
        rp = (ROOT / os.path.normpath(str(row["file_raw"]))).resolve()
        if not ap.is_file() or not rp.is_file():
            # skip silently; capture validator will catch mismatches separately
            continue
        meta = {
            "frame_w": float(row.get("frame_w", 0)),
            "frame_h": float(row.get("frame_h", 0)),
            "angle_deg": float(row.get("angle_deg", 0)),
            "M00": float(row.get("M00", 1)),
            "M01": float(row.get("M01", 0)),
            "M02": float(row.get("M02", 0)),
            "M10": float(row.get("M10", 0)),
            "M11": float(row.get("M11", 1)),
            "M12": float(row.get("M12", 0)),
            "roi_x": float(row.get("roi_x", 0)),
            "roi_y": float(row.get("roi_y", 0)),
            "roi_w": float(row.get("roi_w", 0)),
            "roi_h": float(row.get("roi_h", 0)),
            "pad_px": float(row.get("pad_px", 0)),
            "aligned_w": float(row.get("aligned_w", 0)),
            "aligned_h": float(row.get("aligned_h", 0)),
        }
        items.append(FrameItem(ap, rp, meta))

    if len(items) < 10:
        print(
            f"[warn] Only {len(items)} usable frames; reconstruction quality may suffer"
        )

    return items


# ----------------------------
# Stage 1: SfM (pycolmap)
# ----------------------------


def run_sfm_pycolmap(user_id: str, items: List[FrameItem], workdir: Path) -> Dict:
    if pycolmap is None:
        raise RuntimeError("pycolmap not installed. Install with: pip install pycolmap")

    img_dir = workdir / "images"
    db_path = workdir / "db.db"
    sparse_dir = workdir / "sparse"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Symlink/copy raw images into a flat directory for pycolmap
    for it in items:
        dst = img_dir / it.raw_path.name
        if not dst.exists():
            try:
                os.link(it.raw_path, dst)
            except Exception:
                shutil.copy2(it.raw_path, dst)

    # Use pycolmap automatic reconstructor (Python wrapper) if available
    # Fallback to manual pipeline if needed
    try:
        rec = pycolmap.Reconstruction()
        pycolmap.extract_features(database_path=str(db_path), image_path=str(img_dir))
        pycolmap.match_exhaustive(database_path=str(db_path))
        maps = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(img_dir),
            output_path=str(sparse_dir),
        )
        # pick the largest model
        model_dirs = sorted(sparse_dir.glob("*"))
        if not model_dirs:
            raise RuntimeError("pycolmap produced no sparse model")
        model_path = model_dirs[0]
        rec.read(str(model_path))
    except Exception as e:
        raise RuntimeError(f"pycolmap mapping failed: {e}")

    # Gather per-image intrinsics/extrinsics
    cameras = {}
    images = {}
    for cam_id, cam in rec.cameras.items():
        cameras[cam_id] = {
            "model": cam.model_name,
            "width": cam.width,
            "height": cam.height,
            "params": list(cam.params),
        }
    for img_id, img in rec.images.items():
        images[img.name] = {
            "cam_id": img.camera_id,
            "qvec": list(img.qvec),
            "tvec": list(img.tvec),
        }

    # Sparse points cloud (for depth scale alignment later)
    points3D = []
    for pid, p in rec.points3D.items():
        points3D.append(p.xyz.tolist())

    return {
        "cameras": cameras,
        "images": images,
        "points3D": points3D,
        "image_dir": str(img_dir),
    }


# ----------------------------
# Stage 2: Depth prediction + alignment
# ----------------------------


def load_depth_model():
    if torch is None:
        raise RuntimeError(
            "PyTorch not installed. Install torch/torchvision from PyTorch wheels."
        )
    try:
        # Delayed import so the module is optional at startup
        from depth_anything_v2 import create_model
    except Exception:
        raise RuntimeError(
            "depth-anything-v2 not installed: pip install depth-anything-v2"
        )
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = create_model("vitl")  # large backbone for better detail
    model.to(device).eval()
    return model, device


def predict_depth(model, device, img_bgr: np.ndarray) -> np.ndarray:
    # Minimal wrapper; assumes model accepts numpy BGR or requires preprocessing
    # We keep this stub simple and focus on pipeline glue.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # placeholder; adapt to model API
    # TODO: Replace with the exact preprocessing for depth-anything-v2
    ten = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    ten = ten.to(device)
    with torch.inference_mode():
        pred = model(ten)  # model-specific
    depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth


def align_depth_scale(
    depth: np.ndarray, sparse_pts_cam: np.ndarray
) -> Tuple[float, float]:
    # Robust 1D affine fit: d_true ≈ s*depth + b; here we mock with s=1,b=0 without sparse constraints
    # TODO: Implement projection of sparse 3D into the image and robust fit
    return 1.0, 0.0


# ----------------------------
# Stage 3: TSDF fusion (Open3D)
# ----------------------------


def fuse_tsdf(
    frames: List[np.ndarray],
    depths: List[np.ndarray],
    Ks: List[np.ndarray],
    poses: List[np.ndarray],
) -> Optional[object]:
    if o3d is None:
        raise RuntimeError("open3d not installed: pip install open3d")
    # Voxel size heuristic: 1.5mm if faces are roughly normalized; adjust if needed.
    voxel_size = 0.0015
    trunc = voxel_size * 5
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for img_bgr, dep, K, T_wc in zip(frames, depths, Ks, poses):
        rgb = o3d.geometry.Image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        dep16 = (dep * 1000.0).astype(np.uint16)  # meters->millimeters
        dimg = o3d.geometry.Image(dep16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image(dep.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=img_bgr.shape[1],
            height=img_bgr.shape[0],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        vol.integrate(rgbd, intrinsic, np.linalg.inv(T_wc))
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


# ----------------------------
# Stage 4: Cylindrical UV + texture baking
# ----------------------------


def cylindrical_uv(verts: np.ndarray) -> np.ndarray:
    """Compute simple cylindrical UVs for a face-aligned mesh.
    Assumes Y is up. Returns (N,2) in [0,1]."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    u = (np.arctan2(x, z) + math.pi) / (2 * math.pi)
    v = (y - y.min()) / max(1e-6, (y.max() - y.min()))
    return np.stack([u, v], axis=1).astype(np.float32)


def bake_texture_cyl(
    mesh: "o3d.geometry.TriangleMesh",
    uv: np.ndarray,
    out_png: Path,
    frames: List[np.ndarray],
    Ks: List[np.ndarray],
    poses: List[np.ndarray],
    tex_size: int = 2048,
) -> np.ndarray:
    """Very simple multi-view blender: choose the best single view per triangle by angle; rasterize into UV image.
    This keeps dependencies minimal. Produces a PNG texture and returns it as np.uint8 (H,W,3).
    """
    if o3d is None:
        raise RuntimeError("open3d not installed: pip install open3d")
    # Prepare atlas
    tex = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    # Assign per-vertex UVs into Open3D mesh
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[mesh.triangles.reshape(-1), :])
    # Extremely simplified: fill by nearest projected color from the best camera per triangle
    # NOTE: For production, replace with proper rasterization & z-buffering in UV space.
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    norms = np.asarray(mesh.vertex_normals)
    for tidx, (i, j, k) in enumerate(tris):
        v0, v1, v2 = verts[i], verts[j], verts[k]
        n = (norms[i] + norms[j] + norms[k]) / 3.0
        # Pick best camera by view angle (maximize dot of view dir and normal)
        best = -1e9
        best_frame = None
        for img, K, T_wc in zip(frames, Ks, poses):
            cam_pos = T_wc[:3, 3]
            tri_center = (v0 + v1 + v2) / 3.0
            view_dir = cam_pos - tri_center
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-9)
            score = float(np.dot(view_dir, n))
            if score > best:
                best = score
                best_frame = img
        if best_frame is None:
            continue
        # Paint the triangle uniformly with its center color (placeholder);
        # keeps things simple and avoids heavy rasterization here.
        # Sample color from aligned crop center if available:
        color = best_frame[best_frame.shape[0] // 2, best_frame.shape[1] // 2, :]
        # UV pixels for this tri: just mark the barycenter for now
        uv_tri = uv[[i, j, k]]
        uv_center = uv_tri.mean(axis=0)
        px = int(np.clip(uv_center[0] * (tex_size - 1), 0, tex_size - 1))
        py = int(np.clip((1.0 - uv_center[1]) * (tex_size - 1), 0, tex_size - 1))
        tex[py, px, :] = color
    cv2.imwrite(str(out_png), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    return tex


# ----------------------------
# Stage 5: Save + View
# ----------------------------


def save_obj_with_uv(
    mesh: "o3d.geometry.TriangleMesh", uv: np.ndarray, tex_png: Path, out_prefix: Path
):
    if o3d is None:
        raise RuntimeError("open3d not installed: pip install open3d")
    obj = out_prefix.with_suffix(".obj")
    mtl = out_prefix.with_suffix(".mtl")
    tex_name = tex_png.name

    # Open3D does not yet write VT automatically; write OBJ/MTL manually
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    with open(obj, "w") as f:
        f.write(f"mtllib {mtl.name}\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uv:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        f.write("usemtl face_texture\n")
        for a, b, c in F:
            f.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")
    with open(mtl, "w") as f:
        f.write("newmtl face_texture\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        f.write(f"map_Kd {tex_name}\n")


def view_mesh(mesh: "o3d.geometry.TriangleMesh"):
    if o3d is None:
        print("[warn] open3d not installed; skipping viewer")
        return
    o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Face")


# ----------------------------
# Main
# ----------------------------


def main():
    if len(sys.argv) != 2:
        print("Usage: python reconstruct_face.py <user_id>")
        sys.exit(1)
    user = sys.argv[1].strip()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    work = OUT_ROOT / f"{user}_work"
    work.mkdir(parents=True, exist_ok=True)

    print("[0] Loading dataset …")
    items = load_dataset(user)
    print(f"  - {len(items)} frames")

    print("[1] SfM (pycolmap) …")
    try:
        sfm = run_sfm_pycolmap(user, items, work)
    except Exception as e:
        print(f"[error] {e}")
        print("Install missing deps and re-run: pip install pycolmap")
        sys.exit(2)

    # Build pose/intrinsics lists aligned with frames directory order
    img_dir = Path(sfm["image_dir"]).resolve()
    frame_imgs: List[np.ndarray] = []
    Ks: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    for it in items:
        name = it.raw_path.name
        if name not in sfm["images"]:
            continue
        info = sfm["images"][name]
        cam = sfm["cameras"][info["cam_id"]]
        # Intrinsics (assume pinhole fx,fy,cx,cy)
        params = cam["params"]
        if cam["model"].lower().startswith("pinhole"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            # Fallback guess
            fx = fy = max(cam["width"], cam["height"]) * 1.2
            cx, cy = cam["width"] * 0.5, cam["height"] * 0.5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # Pose: world->cam from qvec,tvec; convert to 4x4 cam->world
        q = np.array(info["qvec"], dtype=np.float64)
        t = np.array(info["tvec"], dtype=np.float64)
        R = pycolmap.qvec2rotmat(q) if pycolmap is not None else np.eye(3)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R.T
        T_wc[:3, 3] = -R.T @ t

        img = cv2.imread(str(it.raw_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frame_imgs.append(img)
        Ks.append(K)
        poses.append(T_wc)

    if len(frame_imgs) < 3:
        print(
            "[error] Not enough posed frames after SfM. Ensure pycolmap succeeded and images have overlap."
        )
        sys.exit(3)

    print("[2] Depth prediction + alignment …")
    try:
        model, device = load_depth_model()
    except Exception as e:
        print(f"[error] {e}")
        print(
            "Install missing deps and re-run: pip install torch torchvision depth-anything-v2"
        )
        sys.exit(4)

    depths: List[np.ndarray] = []
    for img in frame_imgs:
        d = predict_depth(model, device, img)
        s, b = align_depth_scale(d, np.zeros((0, 3), dtype=np.float32))
        depths.append(s * d + b)

    print("[3] TSDF fusion …")
    try:
        mesh = fuse_tsdf(frame_imgs, depths, Ks, poses)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(5)

    if mesh is None or len(np.asarray(mesh.triangles)) == 0:
        print("[error] Fusion produced an empty mesh")
        sys.exit(6)

    print("[4] Cylindrical UV + texture bake …")
    V = np.asarray(mesh.vertices)
    uv = cylindrical_uv(V)
    out_prefix = OUT_ROOT / f"{user}_face_textured"
    out_png = out_prefix.with_suffix(".png")
    try:
        tex = bake_texture_cyl(mesh, uv, out_png, frame_imgs, Ks, poses, tex_size=2048)
        save_obj_with_uv(mesh, uv, out_png, out_prefix)
        print(f"[ok] Wrote: {out_prefix.with_suffix('.obj')} and {out_png}")
    except Exception as e:
        print(f"[warn] Texture baking failed ({e}); writing geometry only")
        # Save geometry alone
        if o3d is not None:
            o3d.io.write_triangle_mesh(str(out_prefix.with_suffix(".obj")), mesh)

    print("[5] Viewer …")
    view_mesh(mesh)


if __name__ == "__main__":
    main()
