"""
face_registration.py

Captures face images via webcam and stores them for later processing.
"""

import cv2
import cv2.data
import os
import time
import uuid

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_crop = frame[y : y + h, x : x + w]
            filename = os.path.join(user_dir, f"{user_id}_{str(uuid.uuid4())}.jpg")
            cv2.imwrite(filename, face_crop)
            count += 1
            print(f"Captured image {count}/{num_images} for user ID: {user_id}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Face capture interrupted by user.")
            break

        time.sleep(0.5)  # Slight delay to avoid rapid captures

    print(f"Finished capturing images for user ID: {user_id}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Face Registration Module")
    user_id_input = input("Enter User ID for face registration: ")
    num_images_input = int(
        input("Enter number of images to capture (default 10): ") or 10
    )
    capture_face_images(user_id_input, num_images_input)
