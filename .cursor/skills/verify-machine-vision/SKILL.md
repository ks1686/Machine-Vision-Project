---
name: verify-machine-vision
description: Drive the Machine Vision Project face registration, authentication, comparison, storage, and mesh viewer CLIs and capture proof. Use when verifying face_registration.py, authenticate.py, compare_faces.py, secure_storage.py, mesh_textured.py, or mesh_viewer.py.
---

# Verify Machine Vision

The user-facing surface is a set of Python CLIs plus OpenCV and Open3D windows. There is no web UI. `uv run python <script>` is the user command. This skill launches an isolated copy of the repo, drives those same entry points, and keeps enrolled faces inside the run.

A machine with no `/dev/video0` cannot open the webcam the scripts request. On Linux the launch compiles `scripts/fakevideo.c` into the run's private `bin/libfakevideo.so` and the drives point `FAKEVIDEO_PPM` at a synthetic face. That preload is verification scaffolding. A checkout with a real webcam does not compile or load it. On Darwin the harness never compiles that shim and never sets `DYLD_INSERT_LIBRARIES`. It opens camera index 0 with AVFoundation. The frames on Linux come from `scripts/synthetic_face.py`. They are drawn shapes, not a photograph of a person.

## Launch

From the repo root, with `uv` on `PATH`:

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision launch --repo "$PWD"
```

Stdout is three lines: `run_id=`, `state=`, and `evidence=`. Ready means `doctor --id RUN_ID` prints `doctor: ok`. Launch copies only `git ls-files` into the state sandbox, records handler defaults before any child starts, and runs `uv sync --frozen --group dev` when `pyproject.toml` and `uv.lock` are present. The app home is `state/home`. Tool caches recorded at launch are passed through. `GOTELEMETRY=off`.

On Linux without `/dev/video0`, launch compiles the camera shim before it prints ready. On Darwin, launch does not compile a shim. A later command sets `CFFIXED_USER_HOME` to the isolated home. Cleanup of a failed launch removes the new state and evidence directories.

## Doctor

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision doctor --id RUN_ID
```

Doctor checks that the state and evidence directories are private, that `handlers.before` exists, and, when the project is present, that `cv2` and `mediapipe` import and that one camera frame has a real height and width. On Linux it prints `camera=device` when `/dev/video0` exists and `camera=shim` when the private shim supplies the frame. On Darwin it opens camera index 0 with `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`. A real frame prints `camera=device` and `doctor: ok`. If that open fails, doctor exits 3 and prints `Darwin has no /dev/video0 camera node and this harness will not inject a library`. It does not say `camera shim missing`. It does not enroll a face.

## Drive

Drive one feature at a time. The user id is `synth_a`. Registration has to succeed before authenticate, compare, storage, or the mesh viewer.

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID register-face
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID view-mesh
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID authenticate
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID compare-faces
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID secure-storage
```

Each drive uses `uv run --frozen python -u` on the real script. GUI scripts get a new session. The harness sends SPACE with `xdotool` to the window whose title matches exactly, and it grabs that window with `ffmpeg`. The SPACE helper stops when its deadline passes, and a timed-out drive reaps that helper instead of waiting on it. A drive returns the wrapped process status. On Darwin a camera that does not open returns 3, the same code as doctor. `authenticate.py` still exits 0 when the face is rejected, so a wrong printed verdict makes the drive return 1 and the transcript keeps the raw `exit_code`.

`run` is the escape hatch for a command inside the sandbox:

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision run --id RUN_ID -- false
```

It returns that command's status. `tel:`, `sms:`, `mailto:`, `sendmail`, and `git push` are recorded and not executed.

## Evidence

Proof files land in the evidence directory from launch, mode `600`. Keep the transcript and the screenshot for the action, not only the last line. Registration proof is the capture window, `Guided continuous capture complete`, `check_capture.py`, and `make_average_face.py`. Authentication proof is the accept window and the reject window with the printed confidence. Storage proof is `storage-at-rest.txt` plus the edit and corrupt transcripts. The mesh proof is the Open3D window titled `synth_a_face.obj`.

## Cleanup

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision cleanup --id RUN_ID
```

Cleanup re-reads handler defaults into `handlers.after` by writing a temp file and renaming it, so a retry replaces the previous snapshot. It signals the recorded process only when the live pid, command, uid, and UTC start time still match, then removes the state directory. It prints `cleaned` and `evidence=<path>`. The evidence directory stays. On a mismatch or a `ps` failure it exits nonzero, prints `cleanup incomplete; state kept`, and prints `recovery:`. A second `cleanup --id` after the fault is fixed removes the state.

```bash
.cursor/skills/verify-machine-vision/scripts/verify-machine-vision recover --id RUN_ID
```

Recover removes the state directory, including enrolled templates and `libfakevideo.so`, and writes `recover.txt` in the evidence directory. It prints `reclaimed` when that state is gone and no recorded process is still alive. If the state directory remains, or a recorded process is still running because its identity did not match, it prints `recover incomplete`, leaves the evidence, and exits nonzero.

## Helpers

- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision launch --repo <repo>`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision doctor --id RUN_ID`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID register-face`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID authenticate`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID compare-faces`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID secure-storage`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision drive --id RUN_ID view-mesh`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision run --id RUN_ID -- <command>`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision cleanup --id RUN_ID`
- `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision recover --id RUN_ID`
- `.cursor/skills/verify-machine-vision/scripts/regression.sh`

`scripts/guard.py` is the path and snapshot checker the shell calls. `scripts/fakevideo.c` and `scripts/synthetic_face.py` are the camera scaffolding above. The regression script proves the harness checks and removes its own scratch.

## Unverified without a webcam or a Mac display

- A real webcam on Linux. This environment has no video device, so the shim supplies frames.
- A real Mac camera. Darwin calls `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)` and exits 3 when that open fails. This VM cannot open an AVFoundation device, so the success path is unverified here.
- The depth-variance reject. The synthetic face is above the 0.003 variance gate. A flat frame was not driven.
- The text `Authentication failed - Wrong identity`. With a target user, `verify_face` only scores that user, so a different face is `No match found`.
- macOS window capture. Darwin runs set `CFFIXED_USER_HOME` and leave the real preference files alone. They do not look up windows.
- A physical key press. SPACE is an `xdotool` event to the exact window title.
- Package install. Open3D needs `libEGL.so.1`. The Matplotlib fallback needs `tkinter`. The harness does not apt-install either.

## Maintenance

Use `/maintain-verification-skill` when a CLI, window title, threshold, or storage rule changes. Update the feature file and re-drive that feature. Do not weaken a journey to fit a stand-in.
