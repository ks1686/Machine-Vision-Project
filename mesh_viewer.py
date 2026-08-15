# mesh_viewer.py
# View the 3D face model built by mesh_textured.py.
# Usage:
#   python mesh_viewer.py ks1686
#   python mesh_viewer.py ks1686 --file models/ks1686_face.obj

from __future__ import annotations

import argparse
import os

import numpy as np


def _pick_default_paths(user_id: str, models_dir: str):
    # Prefer textured OBJ, then triangulated ply, then alpha ply, then points, then numpy
    textured = os.path.join(models_dir, f"{user_id}_face.obj")
    tri = os.path.join(models_dir, f"{user_id}_mesh_delaunay.ply")
    alpha = os.path.join(models_dir, f"{user_id}_mesh_alpha.ply")
    points = os.path.join(models_dir, f"{user_id}_mesh_points.ply")
    npy = os.path.join(models_dir, f"{user_id}_mesh_mean.npy")
    return [textured, tri, alpha, points, npy]


# ---------------- Open3D path ----------------


def _o3d_view(file_path: str, flip_view: bool):
    import os

    import cv2
    import open3d as o3d

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)
        if not mesh.has_vertices() or (not mesh.has_triangles()):
            raise RuntimeError(f"OBJ has no geometry: {file_path}")

        needs_tex = not (hasattr(mesh, "textures") and len(mesh.textures) > 0)
        if needs_tex:
            tex_guess = os.path.join(
                os.path.dirname(file_path),
                os.path.basename(file_path).replace("_face.obj", "_face_texture.png"),
            )
            if os.path.exists(tex_guess) and mesh.has_triangle_uvs():
                img = cv2.imread(tex_guess, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mesh.textures = [o3d.geometry.Image(img)]

        if flip_view:
            R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0))
            mesh.rotate(R, center=(0.0, 0.0, 0.0))

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries(
            [mesh], window_name=os.path.basename(file_path)
        )
        return

    if ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh.has_vertices() and mesh.has_triangles():
            if flip_view:
                R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0))
                mesh.rotate(R, center=(0.0, 0.0, 0.0))
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh], window_name=os.path.basename(file_path)
            )
            return
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise RuntimeError(f"PLY has no geometry: {file_path}")
        if flip_view:
            R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0))
            pcd.rotate(R, center=(0.0, 0.0, 0.0))
        o3d.visualization.draw_geometries(
            [pcd], window_name=os.path.basename(file_path)
        )
        return

    if ext == ".npy":
        pts = np.load(file_path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise RuntimeError(f"NPY is not an Nx3 point cloud: {file_path}")
        if flip_view:
            pts = pts.copy()
            pts[:, 0] *= -1.0
            pts[:, 2] *= -1.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        o3d.visualization.draw_geometries(
            [pcd], window_name=os.path.basename(file_path)
        )
        return

    raise RuntimeError(f"Unsupported file for Open3D viewer: {file_path}")


# ---------------- Matplotlib fallback ----------------


def _parse_obj_vertices_faces(path: str):
    v = []
    f = []
    with open(path, "r") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                v.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = []
                for p in parts:
                    # formats: v, v/t, v//n, v/t/n
                    i = p.split("/")[0]
                    idx.append(int(i) - 1)
                if len(idx) == 3:
                    f.append(tuple(idx))
                else:
                    # fan triangulate if polygon
                    for k in range(1, len(idx) - 1):
                        f.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


def _mpl_view(file_path: str, flip_view: bool):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".obj":
        V, F = _parse_obj_vertices_faces(file_path)
        if V.size == 0 or F.size == 0:
            raise RuntimeError(f"Empty OBJ: {file_path}")
        if flip_view:
            V[:, 0] *= -1.0
            V[:, 2] *= -1.0
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tris = V[F]
        ax.add_collection3d(
            Poly3DCollection(
                tris, facecolors="#8ecae6", edgecolors="#023047", linewidths=0.2, alpha=0.9
            )
        )
        mins, maxs = V.min(axis=0), V.max(axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_title(os.path.basename(file_path))
    elif ext == ".npy":
        pts = np.load(file_path).astype(np.float32)
        if flip_view:
            pts[:, 0] *= -1.0
            pts[:, 2] *= -1.0
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, depthshade=True)
        ax.set_title(os.path.basename(file_path))
    elif ext == ".ply":
        # very simple PLY (vertex-only) reader
        pts = []
        faces = []
        n_vertices = 0
        n_faces = 0
        header = True
        with open(file_path, "r") as f:
            for line in f:
                if header:
                    if line.startswith("element vertex"):
                        n_vertices = int(line.split()[-1])
                    elif line.startswith("element face"):
                        n_faces = int(line.split()[-1])
                    elif line.strip() == "end_header":
                        header = False
                    continue
                if n_vertices > 0:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        n_vertices -= 1
                        continue
                elif n_faces > 0:
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[0] in {"3", "3.0"}:
                        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                    n_faces -= 1

        pts = np.asarray(pts, np.float32)
        V = pts
        if flip_view and V.size:
            V[:, 0] *= -1.0
            V[:, 2] *= -1.0
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        if faces:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            face_idx = np.asarray(faces, np.int32)
            ax.add_collection3d(
                Poly3DCollection(
                    V[face_idx],
                    facecolors="#8ecae6",
                    edgecolors="#023047",
                    linewidths=0.2,
                    alpha=0.9,
                )
            )
            mins, maxs = V.min(axis=0), V.max(axis=0)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
        else:
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=3, depthshade=True)
        ax.set_title(os.path.basename(file_path))
    else:
        raise RuntimeError(f"Unsupported file for Matplotlib fallback: {file_path}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    # Set a helpful view
    ax.view_init(elev=10, azim=70)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="View the 3D face model (OBJ mesh, triangulated PLY, point cloud, or NPY)."
    )
    ap.add_argument("user_id")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--file", default=None, help="Explicit .obj/.ply/.npy path to view")
    ap.add_argument(
        "--no_flip",
        action="store_true",
        help="Do not apply 180° Y-rotation in the viewer",
    )
    args = ap.parse_args()

    if args.file:
        path = args.file
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    else:
        candidates = _pick_default_paths(args.user_id, args.models_dir)
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        else:
            raise FileNotFoundError(
                f"No outputs found in {args.models_dir} for user {args.user_id}."
            )

    flip_view = not args.no_flip

    # Try Open3D first
    try:
        _o3d_view(path, flip_view=flip_view)
        return
    except Exception as e:
        print(
            f"[mesh_viewer] Open3D unavailable or failed ({e}). Falling back to Matplotlib."
        )

    # Fallback
    _mpl_view(path, flip_view=flip_view)


if __name__ == "__main__":
    main()
