"""Candidate selection and outlier filtering for the averaged face."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import make_average_face


def _write_capture(base, user_id, name, blur):
    user_dir = base / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / name).write_bytes(b"png")
    return {"user_id": user_id, "file": f"{user_id}/{name}", "blur_var": blur}


def test_numeric_user_id_matches_metadata(tmp_path):
    rows = [
        _write_capture(tmp_path, "1234", "1234_fid_r00_s01_aa.png", 50.0),
        _write_capture(tmp_path, "1234", "1234_fid_r00_s01_bb.png", 80.0),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)

    df = make_average_face._collect_candidates("1234", str(tmp_path), step=1)

    assert len(df) == 2
    assert df["blur_var"].tolist() == [80.0, 50.0]


def test_keep_raises_clear_error_when_nothing_aligned():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No frames could be aligned"):
        make_average_face._keep_by_reprojection_error([(img, float("inf"))] * 3)


def test_keep_drops_reprojection_outlier():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    errs = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 50.0]
    kept = make_average_face._keep_by_reprojection_error([(img, e) for e in errs])
    assert 50.0 not in [e for _, e in kept]
    assert len(kept) >= 6
