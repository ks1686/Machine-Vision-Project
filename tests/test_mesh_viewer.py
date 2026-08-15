"""Unit tests for mesh viewer fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import mesh_viewer


def test_o3d_view_raises_for_unknown_extension(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"not-a-mesh")

    with pytest.raises(RuntimeError, match="Unsupported"):
        mesh_viewer._o3d_view(str(path), flip_view=False)
