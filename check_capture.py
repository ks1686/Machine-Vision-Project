# check_capture.py

import os
import sys
import cv2
import pandas as pd

OUTPUT_SIZE = (512, 512)  # keep in sync with face_processing.py


def main(user):
    root = "registered_faces"
    user_dir = os.path.join(root, user)
    raw_dir = os.path.join(user_dir, "raw")
    meta_path = os.path.join(root, "metadata.csv")

    assert os.path.isdir(user_dir), f"Missing dir: {user_dir}"
    assert os.path.isdir(raw_dir), f"Missing dir: {raw_dir}"
    assert os.path.isfile(meta_path), f"Missing metadata: {meta_path}"

    # Collect files
    aligned = sorted([f for f in os.listdir(user_dir) if f.endswith(".png")])
    raw = sorted([f for f in os.listdir(raw_dir) if f.endswith(".png")])
    print(f"Found {len(aligned)} aligned PNGs, {len(raw)} raw PNGs")

    # Metadata rows for this user
    df = pd.read_csv(meta_path)
    dfu = df[df["user_id"] == user].copy()
    print(f"Metadata rows for {user}: {len(dfu)}")

    # Expect one metadata row per aligned (accepted) capture
    assert len(dfu) == len(aligned) == len(raw), (
        "Counts mismatch (aligned/raw/metadata)"
    )

    # Verify schema columns exist
    required_cols = [
        "file",
        "file_raw",
        "frame_w",
        "frame_h",
        "aligned_w",
        "aligned_h",
        "angle_deg",
        "M00",
        "M01",
        "M02",
        "M10",
        "M11",
        "M12",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "pad_px",
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns in metadata: {missing}"

    # Verify each file path and sizes
    ok = 0
    for _, row in dfu.iterrows():
        aligned_path = os.path.join(root, os.path.normpath(row["file"]))
        raw_path = os.path.join(root, os.path.normpath(row["file_raw"]))
        assert os.path.isfile(aligned_path), f"Missing aligned: {aligned_path}"
        assert os.path.isfile(raw_path), f"Missing raw: {raw_path}"

        a = cv2.imread(aligned_path, cv2.IMREAD_UNCHANGED)
        r = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        assert a is not None and r is not None, "Failed to read images"

        ah, aw = a.shape[:2]
        assert (ah, aw) == OUTPUT_SIZE, (
            f"Aligned size mismatch: {(ah, aw)} != {OUTPUT_SIZE}"
        )

        rh, rw = r.shape[:2]
        assert int(row["frame_w"]) == rw and int(row["frame_h"]) == rh, (
            "Frame size mismatch in metadata"
        )

        # Basic ROI sanity
        rx, ry, rw_meta, rh_meta = (
            int(row["roi_x"]),
            int(row["roi_y"]),
            int(row["roi_w"]),
            int(row["roi_h"]),
        )
        assert 0 <= rx < rw and 0 <= ry < rh, "ROI origin out of bounds"
        assert (
            rw_meta > 0 and rh_meta > 0 and rx + rw_meta <= rw and ry + rh_meta <= rh
        ), "ROI size invalid"

        ok += 1

    print(f"✓ Passed {ok} captures. Flow looks good.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_capture.py <user_id>")
        sys.exit(1)
    main(sys.argv[1])
