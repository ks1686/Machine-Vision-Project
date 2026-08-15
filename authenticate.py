"""
authenticate.py
---------------
Captures a single face image for authentication purposes.
Uses the same face processing pipeline as registration but simplified for one-shot capture.
"""

import os
import sys
import time

import cv2
import numpy as np

from compare_faces import verify_face
from face_processing import align_face, assess_quality, detect_face_and_landmarks

# Reuse same camera index as registration for consistency
CAMERA_INDEX = 0


def capture_auth_image(
    preview_window: bool = True,
) -> tuple[np.ndarray | None, bool]:
    """
    Capture a single high-quality face image for authentication.
    Returns (frame, success)
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, False

    captured = None
    if preview_window:
        cv2.namedWindow("Capture Authentication Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capture Authentication Image", 960, 540)

    print("Position your face in the camera...")
    print("Press SPACE when ready, or 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy() if preview_window else frame
            face_detect = detect_face_and_landmarks(frame)
            quality_ok = False
            if face_detect:
                bbox, _landmarks = face_detect
                quality = assess_quality(frame, bbox)
                quality_ok = quality.passed
                if preview_window:
                    x, y, w, h = bbox
                    color = (0, 255, 0) if quality_ok else (0, 0, 255)
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    status = f"Quality: {quality.blur_var:.1f}"
                    if not quality_ok and quality.reasons:
                        status = f"Rejected: {quality.reasons}"
                    cv2.putText(
                        display,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            if preview_window:
                cv2.imshow("Capture Authentication Image", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and quality_ok:
                captured = frame.copy()
                break
    finally:
        cap.release()
        if preview_window:
            cv2.destroyAllWindows()

    return captured, captured is not None


def main(argv: list[str] | None = None) -> bool:
    os.makedirs("auth_images", exist_ok=True)
    args = sys.argv if argv is None else argv
    user_id = args[1].strip() if len(args) >= 2 else input(
        "Enter User ID to authenticate (or press Enter to detect any user): "
    ).strip()

    print("Starting authentication capture...")
    frame, success = capture_auth_image()

    if not success or frame is None:
        print("Failed to capture a valid face image.")
        return False

    aligned_frame, quality, _meta = align_face(frame, static_image_mode=True)
    if aligned_frame is None:
        reason = quality.reasons or "alignment failed"
        print(f"Failed to align face in authentication image ({reason}).")
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label = user_id if user_id else "unknown"
    auth_path = os.path.join("auth_images", f"auth_{label}_{timestamp}.png")
    aligned_path = os.path.join("auth_images", f"auth_{label}_{timestamp}_aligned.png")

    cv2.imwrite(auth_path, frame)
    cv2.imwrite(aligned_path, aligned_frame)
    print(f"Authentication image saved to: {auth_path}")
    print(f"Aligned image saved to: {aligned_path}")

    # Perform face verification using aligned image
    print("\nComparing face with registered models...")
    matched_user, confidence = verify_face(aligned_frame, user_id if user_id else None)

    if matched_user:
        if user_id and matched_user != user_id:
            print("\n❌ Authentication failed - Wrong identity!")
            print(f"Expected: {user_id}")
            print(f"Detected: {matched_user}")
            print(f"Confidence: {confidence:.1%}")
        else:
            print("\n✅ Authentication successful!")
            print(f"Identified as: {matched_user}")
            print(f"Confidence: {confidence:.1%}")
    else:
        print("\n❌ Authentication failed - No match found")
        if user_id:
            print(f"Could not verify identity as: {user_id}")

    return matched_user is not None


if __name__ == "__main__":
    main()
