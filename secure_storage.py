"""
secure_storage.py
-----------------
User ID validation and owner-only file permissions for biometric data
(captures, averaged faces, meshes, metadata). Kept free of MediaPipe imports.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Sequence

import cv2
import numpy as np

# Letters, digits, "_", "-", "." (not leading), max 64 chars. User IDs become
# directory and file names, so separators and ".." must never get through.
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$")
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def validate_user_id(raw: str) -> str:
    """Normalize spaces to underscores and reject IDs unsafe as path components."""
    user_id = raw.strip().replace(" ", "_")
    if not _USER_ID_RE.fullmatch(user_id) or ".." in user_id:
        raise ValueError(
            f"Invalid user ID {raw!r}: use 1-64 letters, digits, '_', '-', or '.'"
        )
    return user_id


def make_private_dir(path: str) -> None:
    """Create a directory readable only by the current user (POSIX)."""
    os.makedirs(path, mode=PRIVATE_DIR_MODE, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, PRIVATE_DIR_MODE)


def restrict_file(path: str) -> None:
    """Make a file readable and writable only by the current user (POSIX)."""
    with contextlib.suppress(OSError):
        os.chmod(path, PRIVATE_FILE_MODE)


def write_private_image(
    path: str, image: np.ndarray, params: Sequence[int] = ()
) -> bool:
    """cv2.imwrite, then restrict permissions. Returns False if the write failed."""
    if not cv2.imwrite(path, image, list(params)):
        return False
    restrict_file(path)
    return True
