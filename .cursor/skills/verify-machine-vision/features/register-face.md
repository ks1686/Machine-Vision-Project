# Register a face

Registration opens the webcam, walks two rounds of seven poses, and writes aligned captures plus an averaged face for one user id.

## Sub-features

- `register-window` shows `Guided Continuous Capture` and saves frames without a typed command beyond the user id.
- `register-check` runs `check_capture.py` and requires every saved still to be 512 by 512.
- `register-average` writes `<user>_avgface.png` with `make_average_face.py`.
- `register-reject-id` refuses a user id that is not a single safe path component.

## How to get to it (user POV)

- Run `uv run python face_registration.py <USER_ID>`.
- Follow the on-screen pose prompts. SPACE forces a capture. `q` stops early.
- Run `uv run python check_capture.py <USER_ID>`.
- Run `uv run python make_average_face.py <USER_ID>`.

## Driving it with verify-machine-vision

Preconditions:

- `doctor --id RUN_ID` prints `doctor: ok`, `xdotool=present`, and `ffmpeg=present`.
- `DISPLAY` is set.

- **Start capture.** Enroll `synth_a`. Run `verify-machine-vision drive --id RUN_ID register-face`. The window title is exactly `Guided Continuous Capture`.
- **Proof of the window.** The drive writes `register-window.png` in the evidence directory while the prompt is on screen.
- **Saved captures.** `register-transcript.txt` contains `Guided continuous capture complete` and `Total saved:` with a count above zero. `exit_code=0`.
- **Check.** `check-capture.txt` contains `Passed` and `exit_code=0`.
- **Average.** `average-face.txt` contains `[average] wrote:` and `exit_code=0`. `register-new-files.txt` lists `synth_a_avgface.png`.
- **Bad id.** The storage drive covers `../nope`. Registration itself validates the id before it opens the camera.

## Gotchas

- The script hardcodes camera index 0. With no `/dev/video0`, the harness loads the camera shim. A real webcam will not.
- Auto-capture runs about every half second for two rounds. The window stays up for roughly half a minute. Do not kill it when the first still appears.
- Captures and the averaged face are mode `600`. Directories are mode `700`.
- The shim serves RGB24. OpenCV 4.11 treats a BGR24 buffer as one row, and MediaPipe then rejects the frame.
- `register-face` is the only drive that creates the template. Later features fail closed when `synth_a_avgface.png` is missing.
