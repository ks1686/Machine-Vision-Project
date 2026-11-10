# Secure storage

Storage keeps each user's captures in a private directory and refuses a user id that could escape that directory. The image bytes are an ordinary PNG.

## Sub-features

- `storage-mode` leaves the averaged face at mode `600` and the user directory at mode `700`.
- `storage-plaintext` shows the PNG signature. The file is not encrypted.
- `storage-bad-id` rejects `../nope` before the camera opens.
- `storage-edit` still accepts a face after a decodable pixel change. There is no integrity check.
- `storage-corrupt` fails the load when the PNG no longer decodes.

## How to get to it (user POV)

- Register. The averaged face is `registered_faces/<USER_ID>/<USER_ID>_avgface.png`.
- Run `uv run python face_registration.py '../nope'` and read the error.
- Replace or damage the averaged PNG, then run `uv run python authenticate.py <USER_ID>` and press SPACE.

## Driving it with verify-machine-vision

Preconditions:

- `register-face` has completed for this run.
- `doctor --id RUN_ID` prints `doctor: ok`, `xdotool=present`, and `ffmpeg=present`.
- Run this drive after `view-mesh` if you still want the original texture. The corrupt step changes the averaged face.

- **Permissions and header.** Run `verify-machine-vision drive --id RUN_ID secure-storage`. `storage-at-rest.txt` contains `encrypted=no`, `png_magic=yes`, `integrity_mac=absent`, and mode `600`.
- **Bad id.** `storage-invalid-id.txt` contains `Invalid user ID` and `exit_code=1`. The sandbox has no `nope` directory.
- **Pixel edit.** `storage-edit.txt` contains `Authentication successful`. `storage-at-rest.txt` contains `decodable_edit=accepted`.
- **Corrupt PNG.** `storage-corrupt.txt` contains `Failed to load averaged face` and does not contain `Authentication successful`. `storage-at-rest.txt` contains `corrupt_png=rejected_at_decode`.

## Gotchas

- `secure_storage.py` sets permissions. It does not encrypt and it does not sign the image. A green authentication after a pixel edit is the real behavior.
- A CRC-corrupt PNG fails inside image decode, which the comparator prints as `Failed to load averaged face`.
- Invalid ids exit 1 from `face_registration.py` before `VideoCapture`. A valid id that fails authentication still exits 0.
- Do not copy the averaged face out of the run. Cleanup deletes the sandbox.
