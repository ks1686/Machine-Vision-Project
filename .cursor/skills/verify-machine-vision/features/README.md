# Machine vision verification map

This directory is the maintained source for verifying the face CLIs. Read the index, then use the matching feature file as the recipe.

## Baseline preconditions

- `uv` is on `PATH`. The repo has `pyproject.toml` and `uv.lock`.
- Launch with `.cursor/skills/verify-machine-vision/scripts/verify-machine-vision launch --repo <repo>`.
- Run `doctor --id RUN_ID` and require `doctor: ok`. On Darwin, exit 3 means AVFoundation did not open camera 0. The harness will not inject a library there.
- GUI drives need `DISPLAY`, `xdotool`, and `ffmpeg`. Doctor prints whether they are present.
- The enrolled user is `synth_a`. Frames are synthetic. No real person's photo is enrolled.
- Never point a drive at the checkout's own `registered_faces/`. The sandbox inside the run holds the templates.

## Driving conventions

- Start from the baseline unless the feature names an earlier feature.
- Drive `register-face` before `authenticate`, `compare-faces`, `secure-storage`, or `view-mesh`.
- Treat every command as literal. The user id stays `synth_a`.
- Match window titles exactly: `Guided Continuous Capture`, `Capture Authentication Image`, and `synth_a_face.obj`.
- Read the printed verdict. `authenticate.py` exits 0 even when the face is rejected.
- Leave the evidence directory in place. Cleanup removes only the state directory.

## Proof and skip reporting

- Capture the window during the action and the transcript after it.
- CLI proof includes stdout, stderr, and the `exit_code` line in the transcript.
- Storage proof includes the file mode and the PNG header, not a claim the bytes are encrypted.
- Report an unreachable path with the command and the missing precondition.
- Do not mark a skipped entry point verified through a different script.

## Feature entry contract

Each feature file starts with an H1 and one paragraph, then exactly four H2 sections: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify-machine-vision`, and `Gotchas`.

## Features

- [Register a face](./register-face.md) covers guided capture, `check_capture.py`, and the averaged face.
- [Authenticate](./authenticate.md) covers an accept and a reject from the same enrolled user.
- [Compare faces](./compare-faces.md) covers the scores `authenticate.py` prints. There is no compare CLI.
- [Secure storage](./secure-storage.md) covers permissions, plaintext images, a bad user id, a pixel edit, and a corrupt file.
- [View the mesh](./view-mesh.md) covers the textured OBJ and the Open3D window.
