# Authenticate

Authentication opens the webcam, waits until the face passes the quality bar, and compares that frame with the enrolled average after SPACE.

## Sub-features

- `auth-accept` identifies `synth_a` from the frontal synthetic frame.
- `auth-reject` scores the shifted frame at 0 and prints no match.
- `auth-window` shows `Capture Authentication Image` until SPACE and a passing face.

## How to get to it (user POV)

- Run `uv run python authenticate.py synth_a` after registration.
- Put a face in frame until the preview accepts it.
- Press SPACE. `q` quits without a comparison.
- Read the terminal for `Authentication successful` or `Authentication failed - No match found`.

## Driving it with verify-machine-vision

Preconditions:

- `register-face` has completed for this run.
- `doctor --id RUN_ID` prints `doctor: ok`, `xdotool=present`, and `ffmpeg=present`.

- **Accept.** Run `verify-machine-vision drive --id RUN_ID authenticate`. The first capture uses the frontal frame. `authenticate-accept.png` shows `Capture Authentication Image`. `authenticate-accept.txt` contains `Authentication successful` and `Identified as: synth_a`.
- **Reject.** The same drive then captures the shifted frame. `authenticate-reject.png` is the window. `authenticate-reject.txt` contains `Authentication failed - No match found`, `Final confidence: 0.0%`, and `below minimum threshold 78.0%`.
- **Status.** Both transcripts include the process `exit_code`. The process can be 0 on a reject. The drive returns 1 when either verdict text is missing.

## Gotchas

- `__main__` ignores the boolean `main` returns, so a failed authentication still exits 0. Assert on the text.
- SPACE is ignored until quality passes. The harness releases a stuck space key, grabs the window, then repeats SPACE on the exact title. A held space key closes the window before the shot.
- A second synthetic face that only changes color still scores as the same landmarks. The reject frame shifts the center band so the pose difference is above 30 degrees and the score is forced to 0.
- `Wrong identity` is not this journey. The comparator, given `synth_a`, does not search other users.
- The depth gate rejects variance under 0.003. The synthetic accept frame is above that line. A flat spoof was not driven.
