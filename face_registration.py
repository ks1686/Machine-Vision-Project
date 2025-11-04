"""
face_registration.py
--------------------
Webcam capture loop for collecting face images from a user.

Edit points for instructors/students:
- CAMERA_INDEX: Default is 0; change if you have multiple cameras.
- TARGET_COUNT: Default prompt is 10; set a fixed count or pass via CLI if you add argparse.
- PREVIEW_TEXT: Overlay text size/position can be adjusted below.
"""

from __future__ import annotations

import os
import time
import uuid

import cv2
from face_processing import process_frame

# Base output directory for saved face crops and metadata
OUTPUT_DIR = "registered_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If the wrong camera opens, change this to 1 or 2.
CAMERA_INDEX = 0


def capture_face_images(user_id: str, num_images: int = 10) -> None:
    """
    Capture `num_images` aligned face crops for the given user.
    Saves files under registered_faces/<user_id> and appends metadata.csv in the parent directory.
    """
    # Sanitize user_id for filesystem safety (spaces/colons etc.)
    user_id = user_id.strip().replace(" ", "_")
    user_dir = os.path.join(OUTPUT_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting face capture for user ID: {user_id}")
    print("Press 'q' to quit early.")

    count = 0
    num_images = max(1, int(num_images))  # Avoid invalid counts during testing

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            continue

        metrics_text = ""

        # Use backend processor (detects, quality-checks, aligns, saves, and logs metadata)
        base_name = f"{user_id}_{uuid.uuid4().hex}"
        result = process_frame(
            frame_bgr=frame,
            user_id=user_id,
            out_dir=user_dir,
            base_name=base_name,
            return_metrics=True,
        )
        saved_path, q = result if isinstance(result, tuple) else (result, None)

        if saved_path:
            count += 1
            status_text = f"Saved {count}/{num_images}"
            color = (0, 200, 0)
            metrics_text = ""
            print(f"Captured {count}/{num_images}: {saved_path}")
        else:
            if q is not None:
                status_text = f"Rejected: {q.reasons or 'quality too low'}"
                metrics_text = f"blur(Lap) {q.blur_var:.0f} | bright {q.brightness:.0f} | box {q.bbox_ratio:.3f}"
            else:
                status_text = "No valid face detected"
                metrics_text = ""
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        if not saved_path and q is not None:
            cv2.putText(
                frame,
                metrics_text,
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Face capture interrupted by user.")
            break

        time.sleep(0.2)  # Slight delay to avoid rapid captures

    print(f"Finished capturing aligned images for user ID: {user_id}")
    cap.release()
    cv2.destroyAllWindows()


def capture_guided_rotation(
    user_id: str,
    steps: list[tuple[str, float]] | None = None,
    captures_per_step: int = 2,
    hold_seconds: float = 2.0,
    interval_seconds: float = 1,  # increase if users need more time between captures
) -> None:
    """
    Simpler alternative to pose-based guidance: show on-screen prompts and capture a few
    images per prompt. This avoids head-pose estimation entirely and works on any camera.

    Args:
      user_id: identifier for the folder.
      steps: list of (instruction, seconds) tuples. If None, a reasonable default is used.
      captures_per_step: how many images to save during each instruction.
      hold_seconds: how long to display each instruction if steps is None.
      interval_seconds: minimum time between auto-captures.
    """
    safe_user = user_id.strip().replace(" ", "_")
    user_dir = os.path.join(OUTPUT_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    if steps is None:
        # Default sequence loosely mimics FaceID without requiring pose estimation
        steps = [
            ("Look straight ahead", hold_seconds),
            ("Turn head RIGHT", hold_seconds),
            ("Look UP", hold_seconds),
            ("Turn head LEFT", hold_seconds),
            ("Look DOWN", hold_seconds),
            ("Tilt RIGHT ear toward shoulder", hold_seconds),
            ("Tilt LEFT ear toward shoulder", hold_seconds),
        ]

    print("Guided rotation capture starting. Follow the on-screen prompts.")
    print("Press 'q' to quit early.")

    for idx, (instruction, seconds) in enumerate(steps, start=1):
        start = time.time()
        last_capture = 0.0
        saved_for_step = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed = time.time() - start
            remaining = max(0.0, seconds - elapsed)

            # UI overlay
            header = f"Step {idx}/{len(steps)}"
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
                f"saved {saved_for_step}/{captures_per_step}",
                (12, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )

            # Auto-capture at intervals until we have enough for this step
            now = time.time()
            if (
                saved_for_step < captures_per_step
                and (now - last_capture) >= interval_seconds
            ):
                base_name = f"{safe_user}_guided_{idx:02d}_{saved_for_step:02d}"
                result = process_frame(
                    frame_bgr=frame,
                    user_id=safe_user,
                    out_dir=user_dir,
                    base_name=base_name,
                    return_metrics=True,
                )
                saved_path, q = result if isinstance(result, tuple) else (result, None)
                if saved_path:
                    saved_for_step += 1
                    last_capture = now
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
                else:
                    if q is not None:
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
                    last_capture = now

            cv2.imshow("Guided Rotation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Guided rotation capture stopped by user.")
                return

            # Advance to next step when time is up and we have enough captures
            if elapsed >= seconds and saved_for_step >= captures_per_step:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Guided rotation capture complete.")


# For a CLI version, consider using argparse to pass --user, --count, and threshold overrides.
if __name__ == "__main__":
    print("Face Registration Module")
    user_id_input = input("Enter User ID for face registration: ").strip()
    num_raw = input("Enter number of images to capture (default 10): ").strip()
    try:
        num_images_input = int(num_raw) if num_raw else 10
    except ValueError:
        print("Invalid number entered; defaulting to 10.")
        num_images_input = 10

    mode = input("Mode: (s)imple captures, (g)uided rotation [s/g]: ").strip().lower()
    if mode == "g":
        capture_guided_rotation(user_id_input)
    else:
        capture_face_images(user_id_input, num_images_input)
