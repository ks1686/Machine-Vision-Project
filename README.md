# 3D Face Modeling & Authentication Pipeline (MediaPipe + OpenCV)

## Overview

This repository implements face authentication using **only standard RGB webcams** - no specialized depth sensors required.  
It generates a full 3D textured model of a user's face by capturing multiple angles, averaging the results, and building a textured 3D mesh. The system then authenticates users against their registered models using MediaPipe's 478-point facial landmark detection.

**Key Feature**: Achieves face recognition and pose-aware authentication using commodity hardware (any RGB camera), demonstrating that specialized depth sensors like Apple's TrueDepth are not necessary for effective facial authentication.

---

## 1. Prerequisites

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

Then run scripts with `uv run python <script.py> ...`.

Ensure your webcam is connected and functional.

---

## 2. Capture Faces

Use the guided registration tool to collect multiple aligned face captures:

```bash
uv run python face_registration.py <USER_ID>
```

If you omit `<USER_ID>`, the script prompts for one.

Faces are stored in `registered_faces/<USER_ID>/` and automatically aligned using MediaPipe FaceMesh.

---

## 3. Create Average Face

Generate an averaged texture image from all aligned captures:

```bash
uv run python make_average_face.py <USER_ID>
```

Output:  
`registered_faces/<USER_ID>/<USER_ID>_avgface.png`

---

## 4. Build Textured Mesh

Construct a 3D face mesh with the averaged texture mapped to it:

```bash
uv run python mesh_textured.py <USER_ID>
```

Outputs (in `models/`):

- `<USER_ID>_face.obj`
- `<USER_ID>_face.mtl`
- `<USER_ID>_face_texture.png`

---

## 5. View the Model

Visualize the mesh interactively:

```bash
uv run python mesh_viewer.py <USER_ID> --file models/<USER_ID>_face.obj
```

If the model appears flipped, you can disable the default 180° view rotation:

```bash
uv run python mesh_viewer.py <USER_ID> --file models/<USER_ID>_face.obj --no_flip
```

---

## 6. Face Authentication

Once a user is registered and their 3D model is created, you can authenticate them using the authentication system.

### Authenticate a User

Run the authentication script to capture a live image and compare it against registered models:

```bash
uv run python authenticate.py [USER_ID]
```

You'll be prompted to either:
- Press Enter to compare against all registered users
- Enter a specific User ID to verify against that user only

Optional: pass the User ID as the first argument to skip the prompt.

**Authentication Process:**
1. Position your face in the camera frame
2. Press SPACE when ready (or 'q' to quit)
3. The system captures and aligns your face
4. Compares against registered model(s) using 478 facial landmarks
5. Reports authentication result with confidence score

**Security Features:**
- **Pose Detection**: Tracks head rotation (yaw/pitch) and applies penalties for extreme angles
- **Confidence Threshold**: Requires 78% minimum confidence for authentication
- **Facial Landmark Comparison**: Uses MediaPipe's 478-point face mesh for detailed matching

**Expected Results:**
- Registered user (normal pose): typically above the 78% threshold ✅
- Registered user (extreme angle >30°): Rejected ❌
- Different person / failed liveness: 0% reported score (rejected) ❌

Confidence is a linear map of landmark similarity into `[0, 1]`. Extreme pose (`>30°`) and low MediaPipe depth variance fail closed.

**Output Files:**
- `auth_images/auth_<userid>_<timestamp>.png` - Original capture
- `auth_images/auth_<userid>_<timestamp>_aligned.png` - Aligned face image

---

## 7. File Summary

| File                   | Description                                    |
|------------------------|------------------------------------------------|
| `face_processing.py`   | Face detection, alignment, and quality filters |
| `face_registration.py` | Guided webcam capture of multiple poses        |
| `make_average_face.py` | Creates averaged face texture                  |
| `mesh_textured.py`     | Generates 3D mesh with texture                 |
| `mesh_viewer.py`       | Visualizes resulting mesh                      |
| `authenticate.py`      | Live face authentication against registered models |
| `compare_faces.py`     | Face comparison logic with pose detection      |
| `check_capture.py`     | Validates aligned/raw/metadata counts for a user |

Run unit tests with `uv run pytest`. GitHub Actions runs the same suite on every push and pull request (`.github/workflows/test.yml`).

---

## Notes

- The OBJ/MTL/PNG set can be imported into Blender, MeshLab, or any 3D engine for rendering or further processing.