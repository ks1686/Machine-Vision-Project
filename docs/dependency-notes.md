# Dependency constraints: protobuf / MediaPipe

This document records why `protobuf` is pinned below 5.x, why NumPy stays
on 1.x / OpenCV on 4.x / SciPy on 1.17, and why Dependabot ignores those
bumps (see `.github/dependabot.yml`). Revisit after a MediaPipe Tasks-API port.

## The constraint chain

1. **The security advisory has no 4.x fix.**
   GHSA-7gcm-g887-7qv7 / CVE-2026-0994 (high, DoS via recursion-depth bypass in
   `google.protobuf.json_format.ParseDict`) is patched only in **protobuf
   >= 5.29.6**. The newest 4.x release is 4.25.9 and remains vulnerable per the
   advisory range `< 5.29.6`.

2. **MediaPipe 0.10.x declares `protobuf>=4.25.3,<5`.**
   Any protobuf 5.x/6.x/7.x bump cannot resolve alongside it — this is what made
   Dependabot's protobuf security-update job fail repeatedly.

3. **Forcing protobuf >=5 next to MediaPipe 0.10.x breaks at runtime.**
   Verified empirically with mediapipe 0.10.21 + protobuf 5.29.6 installed side
   by side: `FaceMesh` init fails with a libprotobuf error while parsing its
   embedded graph config —

   ```
   [libprotobuf ERROR .../text_format.cc:335] Error parsing text-format
   mediapipe.CalculatorGraphConfig: 68:22: Expected identifier, got: \
   RuntimeError: Failed to parse: node {...}
   ```

   The C++ layer of mediapipe 0.10.x is built against the protobuf 4 ABI /
   text-format behavior; a 5.x runtime is not just undeclared but genuinely
   incompatible.

4. **MediaPipe >= 0.10.30 removed the legacy API this pipeline uses.**
   The whole codebase (`face_processing.py` and everything importing it) is
   built on `mediapipe.python.solutions.face_mesh.FaceMesh` (478 landmarks,
   `refine_landmarks=True`). In 0.10.30+ the package ships only the Tasks API
   (`mediapipe.tasks`); `mediapipe.python.solutions` is gone entirely.
   Migrating is a deliberate port project, not a version bump.

## Why the risk is accepted

The vulnerable function (`json_format.ParseDict`) parses **untrusted protobuf
JSON**. This pipeline never calls it: it deserializes no external protobuf data
at all — it only runs face landmark graphs locally over webcam frames. There is
no reachable attack path from this repository's code.

Accordingly, Dependabot alert #9 (protobuf 4.25.9) was dismissed as
`tolerable_risk` with this document cited as evidence.

## NumPy 2 / OpenCV 5 / SciPy 1.18 (Dependabot PR #8)

Do not take a green CI run on a grouped `uv.lock` bump as proof this stack
moved forward. MediaPipe 0.10.21 declares `numpy<2`. Two other packages in
the same Dependabot group require NumPy 2 on Python 3.12:

| Package | Constraint that collides |
| --- | --- |
| mediapipe 0.10.21 | `numpy<2` (added in 0.10.15; see [google-ai-edge/mediapipe#5612](https://github.com/google-ai-edge/mediapipe/issues/5612)) |
| mediapipe 0.10.14 | `numpy` with **no upper bound** (May 2024, before NumPy 2) |
| opencv-python 5.0.0.93 | `numpy>=2` when `python_version >= "3.9"` |
| scipy 1.18.1 | `numpy>=2.0.0,<2.8` |
| scipy 1.17.1 | `numpy>=1.26.4,<2.7` (compatible with this lock) |

uv will then keep `mediapipe>=0.10.14,<0.10.30` satisfiable by **downgrading**
0.10.21 → 0.10.14 so NumPy 2 can be installed. That is a resolver side-effect,
not a MediaPipe upgrade:

- 0.10.14 never claimed NumPy 2 support; Google added `numpy<2` one release
  later because of graph packet-propagation bugs.
- The advertised OpenCV 5 bump does not even win at runtime. MediaPipe still
  depends on `opencv-contrib-python`, which stays at 4.11.x; both wheels ship
  a `cv2` package, and the imported module remains **4.11.0**.

The pipeline (`face_processing.py` and every importer) uses the legacy
`FaceMesh` API plus OpenCV affine/CLAHE/warp helpers. Unit tests mock webcam
capture and landmark comparison; they import FaceMesh and process a blank
frame, but they do not exercise real faces, alignment scores, or the
OpenCV 5 `warpAffine` numerical change. CI green is not enough to merge a
NumPy 2 / OpenCV 5 move.

`pyproject.toml` therefore pins `numpy<2`, `opencv-python<5`, `scipy<1.18`,
and `mediapipe>=0.10.21,<0.10.30`. Drop those ceilings in the same Tasks-API
port that lifts the protobuf pin.

## Path to actually fixing it

Port `face_processing.py` to the MediaPipe **Tasks API**
(`FaceLandmarker`, 478 landmarks incl. iris), then:

- drop the `protobuf` ignore from `.github/dependabot.yml`,
- drop the NumPy / OpenCV / SciPy ceilings in `pyproject.toml` and the
  matching Dependabot ignores,
- raise `requires-python` if the newer wheels allow,
- re-open/dismiss history for the old alert as fixed.
