"""
face_registration.py
--------------------
FaceID-style guided capture that continuously saves aligned images during each
prompted head pose. There is **one** mode only: guided + continuous capture.

Editable parameters:
- CAMERA_INDEX: choose your camera (0/1/2).
- STEPS: prompts + duration for each step.
- INTERVAL_SECONDS: minimum time between auto-captures.
- ROUNDS: repeat the whole sequence this many times.
"""

from __future__ import annotations

import os
import time
import uuid

import cv2
from face_processing import process_frame

# Output directory
OUTPUT_DIR = "registered_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Camera index (change if the wrong camera opens)
CAMERA_INDEX = 0

# Default FaceID-like prompts
STEPS: list[tuple[str, float]] = [
    ("Look straight ahead", 2.5),
    ("Turn head RIGHT", 2.5),
    ("Look UP", 2.5),
    ("Turn head LEFT", 2.5),
    ("Look DOWN", 2.5),
    ("Tilt RIGHT ear toward shoulder", 2.5),
    ("Tilt LEFT ear toward shoulder", 2.5),
]

# Minimum time between automatic captures
INTERVAL_SECONDS = 0.8

# Repeat the entire sequence this many times
ROUNDS = 1


def run_guided_continuous(
    user_id: str,
    steps: list[tuple[str, float]] | None = None,
    interval_seconds: float = INTERVAL_SECONDS,
    rounds: int = ROUNDS,
) -> None:
    """Guide the user through STEPS and continuously capture aligned images throughout each step."""
    safe_user = user_id.strip().replace(" ", "_")
    user_dir = os.path.join(OUTPUT_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    if steps is None:
        steps = STEPS

    print("Guided continuous capture starting. Follow the on-screen prompts.")
    print("Press 'q' to stop early. Press SPACE to force an immediate capture.")

    total_saved = 0

    for r in range(max(1, int(rounds))):
        for idx, (instruction, seconds) in enumerate(steps, start=1):
            start = time.time()
            last_capture = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                key = cv2.waitKey(1) & 0xFF
                now = time.time()
                elapsed = now - start
                remaining = max(0.0, seconds - elapsed)

                # Overlay UI
                header = f"Step {idx}/{len(steps)} (Round {r + 1})"
                cv2.putText(
                    frame,
                    header,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    instruction,
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"remaining {remaining:0.1f}s",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"total saved {total_saved}",
                    (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Auto-capture at fixed interval; SPACE forces immediate capture
                ready_interval = (now - last_capture) >= interval_seconds
                manual_trigger = key == ord(" ")
                if ready_interval or manual_trigger:
                    base = f"{safe_user}_fid_r{r:02d}_s{idx:02d}_{uuid.uuid4().hex[:8]}"
                    result = process_frame(
                        frame_bgr=frame,
                        user_id=safe_user,
                        out_dir=user_dir,
                        base_name=base,
                        return_metrics=True,
                    )
                    saved_path, q = (
                        result if isinstance(result, tuple) else (result, None)
                    )
                    last_capture = (
                        now  # wait interval even if rejected to avoid spamming
                    )

                    if saved_path:
                        total_saved += 1
                        cv2.putText(
                            frame,
                            "Saved",
                            (12, 132),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    elif q is not None:
                        cv2.putText(
                            frame,
                            f"Rejected: {q.reasons}",
                            (12, 132),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            frame,
                            "No valid face",
                            (12, 132),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow("Guided Continuous Capture", frame)
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Stopped by user. Total saved: {total_saved}")
                    return

                if elapsed >= seconds:
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Guided continuous capture complete. Total saved: {total_saved}")


if __name__ == "__main__":
    print("FaceID-style Guided Continuous Registration")
    user = input("Enter User ID: ").strip()
    run_guided_continuous(user)
