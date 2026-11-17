# Compare faces

Comparison prints pose, depth variance, and confidence for the captured face against the enrolled average. `compare_faces.py` has no command line. The user sees the scores from `authenticate.py`.

## Sub-features

- `compare-score` prints `Comparison for synth_a`, pose, and `Final confidence`.
- `compare-threshold` prints `below minimum threshold 78.0%` when the score is under 0.78.
- `compare-depth` prints `Auth depth variance` before the score.

## How to get to it (user POV)

- Register, then run `uv run python authenticate.py synth_a`.
- Press SPACE when the preview is ready.
- Read the `Comparing face with registered models` section in the terminal.

## Driving it with verify-machine-vision

Preconditions:

- `register-face` has completed for this run.
- `doctor --id RUN_ID` prints `doctor: ok`, `xdotool=present`, and `ffmpeg=present`.

- **Score the frontal frame.** Run `verify-machine-vision drive --id RUN_ID compare-faces`. `compare-window.png` is `Capture Authentication Image`. `compare-transcript.txt` contains `Comparison for synth_a`, `Pose difference:`, `Final confidence:`, `Auth depth variance:`, `Authentication successful`, and `Identified as: synth_a`.
- **Threshold.** The reject half of `authenticate` writes `below minimum threshold 78.0%` and `Final confidence: 0.0%` in `authenticate-reject.txt`. Drive `authenticate` for that artifact. Do not invent a compare subcommand.

## Gotchas

- There is no `python compare_faces.py` user path. Driving a stub of `verify_face` does not count.
- Pose difference above 30 degrees forces the score to 0 before the softer pose penalty.
- The threshold line is printed only when the best score is below 0.78. A successful comparison does not include it.
- The depth reject for variance under 0.003 is a separate branch. This drive does not use a flat image.
