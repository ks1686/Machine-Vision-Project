"""
face_registration.py
--------------------
Webcam capture loop for collecting face images from a user.

Edit points for instructors/students:
- CAMERA_INDEX: Default is 0; change if you have multiple cameras.
- TARGET_COUNT: Default prompt is 10; set a fixed count or pass via CLI if you add argparse.
- PREVIEW_TEXT: Overlay text size/position can be adjusted below.
"""

import cv2
import os
import time
import uuid
from face_processing import process_frame

OUTPUT_DIR = "registered_faces"  # Base directory for saved face crops and metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA_INDEX = 0  # If the wrong camera opens, change this to 1 or 2.


def capture_face_images(user_id: str, num_images: int = 10):
    """
    Capture 'num_images' aligned face crops for the given user.
    Saves files under registered_faces/<user_id> and appends metadata.csv in the parent directory.
    """

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )  # Consider setting resolution: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting face capture for user ID: {user_id}")
    print("Press 'q' to quit early.")

    # Sanitize user_id for filesystem safety (spaces/colons etc.)
    user_id = user_id.strip().replace(" ", "_")
    user_dir = os.path.join(OUTPUT_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

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
            print(f"Captured {count}/{num_images}: {saved_path}")
            status_text = f"Saved {count}/{num_images}"
            color = (0, 200, 0)
        else:
            # Show reasons and current metrics if available
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
    capture_face_images(user_id_input, num_images_input)
