"""
authenticate.py
---------------
Captures a single face image for authentication purposes.
Uses the same face processing pipeline as registration but simplified for one-shot capture.
"""

import os
import cv2
import numpy as np
import time
from face_processing import process_frame, detect_face_and_landmarks, assess_quality
from compare_faces import verify_face

# Reuse same camera index as registration for consistency
CAMERA_INDEX = 0


def capture_auth_image(preview_window: bool = True) -> tuple[np.ndarray, bool]:
    """
    Capture a single high-quality face image for authentication.
    Returns (frame, success)
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Request a more reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    best_frame = None
    best_quality = 0.0

    # Create named window with a specific size
    cv2.namedWindow("Capture Authentication Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Authentication Image", 960, 540)  # 75% of 1280x720

    print("Position your face in the camera...")
    print("Press SPACE when ready, or 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Try to detect face and assess quality
        face_detect = detect_face_and_landmarks(frame)
        if face_detect:
            bbox, landmarks = face_detect
            quality = assess_quality(frame, bbox)

            # Show quality metrics in preview
            if preview_window:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = f"Quality: {quality.blur_var:.1f}"
                cv2.putText(
                    frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

            # Keep track of best frame
            if quality.blur_var > best_quality:
                best_frame = frame.copy()
                best_quality = quality.blur_var

        if preview_window:
            cv2.imshow("Capture Authentication Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # Space to capture
            if best_frame is not None:
                break

    cap.release()
    if preview_window:
        cv2.destroyAllWindows()

    return best_frame, best_frame is not None


def main():
    # Create an auth_images directory if it doesn't exist
    os.makedirs("auth_images", exist_ok=True)

    # Get the user ID to authenticate against
    user_id = input(
        "Enter User ID to authenticate (or press Enter to detect any user): "
    ).strip()

    print("Starting authentication capture...")
    frame, success = capture_auth_image()

    if not success:
        print("Failed to capture a valid face image.")
        return False

    # Align the face like we do during registration
    aligned_path = process_frame(frame, "auth", "auth_images", "auth_capture")
    if not aligned_path:
        print("Failed to align face in authentication image.")
        return False

    # Read the aligned image back
    aligned_frame = cv2.imread(aligned_path)
    if aligned_frame is None:
        print("Failed to read aligned face image.")
        return False

    # Save both original and aligned frames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if user_id:
        auth_path = os.path.join("auth_images", f"auth_{user_id}_{timestamp}.png")
        aligned_path = os.path.join(
            "auth_images", f"auth_{user_id}_{timestamp}_aligned.png"
        )
    else:
        auth_path = os.path.join("auth_images", f"auth_unknown_{timestamp}.png")
        aligned_path = os.path.join(
            "auth_images", f"auth_unknown_{timestamp}_aligned.png"
        )

    cv2.imwrite(auth_path, frame)
    cv2.imwrite(aligned_path, aligned_frame)
    print(f"Authentication image saved to: {auth_path}")
    print(f"Aligned image saved to: {aligned_path}")

    # Perform face verification using aligned image
    print("\nComparing face with registered models...")
    matched_user, confidence = verify_face(aligned_frame, user_id if user_id else None)

    if matched_user:
        if user_id and matched_user != user_id:
            print(f"\n❌ Authentication failed - Wrong identity!")
            print(f"Expected: {user_id}")
            print(f"Detected: {matched_user}")
            print(f"Confidence: {confidence:.1%}")
        else:
            print(f"\n✅ Authentication successful!")
            print(f"Identified as: {matched_user}")
            print(f"Confidence: {confidence:.1%}")
    else:
        print("\n❌ Authentication failed - No match found")
        if user_id:
            print(f"Could not verify identity as: {user_id}")

    return matched_user is not None


if __name__ == "__main__":
    main()
