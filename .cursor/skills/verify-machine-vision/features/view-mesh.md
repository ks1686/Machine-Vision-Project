# View the mesh

The textured mesh build writes an OBJ, an MTL, and a texture from the averaged face. The viewer opens that OBJ in an Open3D window.

## Sub-features

- `mesh-build` writes `models/synth_a_face.obj` and a private texture.
- `mesh-window` opens a window whose title is the file name `synth_a_face.obj`.

## How to get to it (user POV)

- Register, then run `uv run python mesh_textured.py synth_a`.
- Run `uv run python mesh_viewer.py synth_a`.
- Close the window to return to the shell. The Matplotlib fallback is used only if Open3D fails.

## Driving it with verify-machine-vision

Preconditions:

- `register-face` has completed for this run.
- `doctor --id RUN_ID` prints `doctor: ok`, `xdotool=present`, and `ffmpeg=present`.
- `libEGL.so.1` is installed. The harness does not install it.

- **Build.** Run `verify-machine-vision drive --id RUN_ID view-mesh`. `mesh-textured.txt` contains `[mesh_textured] wrote:` and `exit_code=0`. The OBJ is mode `600`.
- **Window.** `mesh-window.png` shows a window titled exactly `synth_a_face.obj`. `mesh-viewer.txt` records `exit_code=signaled` because the viewer blocks until the harness closes that process group.
- **Identity.** The harness signals the process only after the live command, uid, and UTC start time match the process it started.

## Gotchas

- Open3D is the primary viewer. If it cannot load `libEGL.so.1` it prints a fallback line and Matplotlib needs `tkinter`. The drive requires the Open3D title, so a fallback is a failure, not a pass.
- Do not set `MPLBACKEND`. An Agg backend would skip the window.
- The window title is the OBJ basename, not the string `mesh`.
- macOS has no `xdotool` path in this harness. A Darwin run does not claim the window was seen.
