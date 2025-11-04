"""
face_registration.py

Captures face images via webcam and stores them for later processing.
"""

import cv2
import os
import time
import uuid
from face_processing import process_frame

# Directory to save registered face images
OUTPUT_DIR = "registered_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_face_images(user_id: str, num_images: int = 10):
    """
    Captures multiple face images for given user ID via webcam.
    Args:
        user_id (str): Unique identifier for the user.
        num_images (int): Number of face images to capture.
    """

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting face capture for user ID: {user_id}")
    print("Press 'q' to quit early.")

    user_dir = os.path.join(OUTPUT_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    count = 0
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


if __name__ == "__main__":
    print("Face Registration Module")
    user_id_input = input("Enter User ID for face registration: ")
    num_images_input = int(
        input("Enter number of images to capture (default 10): ") or 10
    )
    capture_face_images(user_id_input, num_images_input)
