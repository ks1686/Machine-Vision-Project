# 3D Face Modeling Pipeline (MediaPipe + OpenCV)

## Overview

This repository generates a full 3D textured model of a user's face using standard webcams.  
It captures multiple angles, averages the results, and builds a textured 3D mesh for visualization or export.

---

## 1. Prerequisites

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scipy open3d
```

Ensure your webcam is connected and functional.

---

## 2. Capture Faces

Use the guided registration tool to collect multiple aligned face captures:

```bash
python face_registration.py <USER_ID>
```

Faces are stored in `registered_faces/<USER_ID>/` and automatically aligned using MediaPipe FaceMesh.

---

## 3. Create Average Face

Generate an averaged texture image from all aligned captures:

```bash
python make_average_face.py <USER_ID>
```

Output:  
`registered_faces/<USER_ID>/<USER_ID>_avgface.png`

---

## 4. Build Textured Mesh

Construct a 3D face mesh with the averaged texture mapped to it:

```bash
python mesh_textured.py <USER_ID>
```

Outputs (in `models/`):

- `<USER_ID>_face.obj`
- `<USER_ID>_face.mtl`
- `<USER_ID>_face_texture.png`

---

## 5. View the Model

Visualize the mesh interactively:

```bash
python mesh_viewer.py <USER_ID> --file models/<USER_ID>_face.obj
```

If the model appears flipped, you can disable the default 180° view rotation:

```bash
python mesh_viewer.py <USER_ID> --file models/<USER_ID>_face.obj --no_flip
```

---

## 6. File Summary

| File                   | Description                                    |
|------------------------|------------------------------------------------|
| `face_processing.py`   | Face detection, alignment, and quality filters |
| `face_registration.py` | Guided webcam capture of multiple poses        |
| `make_average_face.py` | Creates averaged face texture                  |
| `mesh_textured.py`     | Generates 3D mesh with texture                 |
| `mesh_viewer.py`       | Visualizes resulting mesh                      |

---

## Notes

- This pipeline is **modeling-only**. Authentication logic (`enroll_geom.py`, `calibrate_geom.py`, `verify_geom.py`) is
  handled separately.
- The OBJ/MTL/PNG set can be imported into Blender, MeshLab, or any 3D engine for rendering or further processing.