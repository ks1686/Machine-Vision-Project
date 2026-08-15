"""
compare_faces.py
---------------
Compares a captured authentication image against registered 3D face models.
Uses MediaPipe Face Mesh landmarks for geometric comparison and feature matching.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from face_processing import facemesh_vector_from_aligned

MIN_CONFIDENCE = 0.78
MIN_DEPTH_VARIANCE = 0.003
MAX_POSE_DIFF_DEG = 30.0
_FEATURE_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def clear_feature_cache() -> None:
    """Drop cached avg-face feature vectors (tests and re-enrollment)."""
    _FEATURE_CACHE.clear()


def load_registered_models(
    models_dir: str = "models", faces_dir: str = "registered_faces"
) -> list[str]:
    """Return user IDs that have an avg-face image and/or a completed OBJ mesh."""
    users: set[str] = set()
    if os.path.isdir(models_dir):
        for name in os.listdir(models_dir):
            if name.endswith("_face.obj"):
                users.add(name[: -len("_face.obj")])
    if os.path.isdir(faces_dir):
        for user_id in os.listdir(faces_dir):
            avg = os.path.join(faces_dir, user_id, f"{user_id}_avgface.png")
            if os.path.isfile(avg):
                users.add(user_id)
    return sorted(users)


def estimate_head_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """
    Estimate head pose angles from 3D landmarks.
    Returns (yaw, pitch) in degrees.
    """
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    left_dist = np.linalg.norm(nose_tip[:2] - left_eye[:2])
    right_dist = np.linalg.norm(nose_tip[:2] - right_eye[:2])
    yaw = (
        np.degrees(np.arctan2(right_dist - left_dist, (left_dist + right_dist) / 2)) * 2
    )

    eye_y = (left_eye[1] + right_eye[1]) / 2
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2
    face_height = mouth_y - eye_y
    if face_height != 0:
        nose_position = (nose_tip[1] - eye_y) / face_height
        pitch = (nose_position - 0.5) * 60
    else:
        pitch = 0.0

    return float(yaw), float(pitch)


def get_face_features(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Extract normalized face mesh features from image.
    Returns (features, depth_variance) or None.
    """
    print(f"- Processing image shape: {image.shape}")

    if image.shape[:2] != (512, 512):
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        print(f"- Resized to: {image.shape}")

    landmarks = facemesh_vector_from_aligned(image)
    if landmarks is None:
        print("- Failed to extract face mesh landmarks")
        return None

    n = landmarks.size // 3
    features = landmarks.reshape(n, 3).astype(np.float32)
    z_variance = float(np.var(features[:, 2]))

    features[:, :2] -= features[:, :2].mean(axis=0, keepdims=True)
    scale = np.sqrt((features[:, :2] ** 2).sum(axis=1)).max()
    if scale > 0:
        features[:, :2] /= scale

    z_range = features[:, 2].max() - features[:, 2].min()
    if z_range > 0:
        features[:, 2] = (features[:, 2] - features[:, 2].min()) / z_range

    return features, z_variance


def get_cached_model_features(
    user_id: str, img_path: str
) -> tuple[np.ndarray, float] | None:
    """Load and cache avg-face features for a registered user."""
    cached = _FEATURE_CACHE.get(user_id)
    if cached is not None:
        return cached
    if not os.path.isfile(img_path):
        return None
    model_img = cv2.imread(img_path)
    if model_img is None:
        print(f"- Failed to load averaged face for user {user_id}")
        return None
    result = get_face_features(model_img)
    if result is None:
        return None
    _FEATURE_CACHE[user_id] = result
    return result


def compare_features(features1: np.ndarray, features2: np.ndarray) -> dict[str, float]:
    """
    Compare two sets of face features, return similarity metrics.
    Score is a linear map of similarity into [0, 1] with a hard pose reject.
    """
    empty = {
        "score": 0.0,
        "base_similarity": 0.0,
        "pose_diff": 0.0,
        "yaw1": 0.0,
        "pitch1": 0.0,
        "yaw2": 0.0,
        "pitch2": 0.0,
    }
    if features1 is None or features2 is None:
        return empty

    if len(features1.shape) == 1:
        features1 = features1.reshape(-1, 3)
    if len(features2.shape) == 1:
        features2 = features2.reshape(-1, 3)

    min_points = min(len(features1), len(features2))
    features1 = features1[:min_points]
    features2 = features2[:min_points]

    yaw1, pitch1 = estimate_head_pose(features1)
    yaw2, pitch2 = estimate_head_pose(features2)
    pose_diff = float(np.sqrt((yaw1 - yaw2) ** 2 + (pitch1 - pitch2) ** 2))

    weights = np.array([1.0, 1.0, 0.5])
    diff = np.sqrt((((features1 - features2) * weights) ** 2).sum(axis=1))
    mean_diff = (np.percentile(diff, 75) + np.percentile(diff, 90)) / 2.0
    z_diff = np.abs(features1[:, 2] - features2[:, 2]).mean()
    base_similarity = float(1.0 / (1.0 + mean_diff + z_diff * 0.5))
    pose_angle_penalty = np.clip(pose_diff / MAX_POSE_DIFF_DEG, 0, 1) * 0.5
    adjusted_similarity = base_similarity * (1.0 - pose_angle_penalty)
    score = float(np.clip((adjusted_similarity - 0.15) / 0.35, 0.0, 1.0))
    if pose_diff > MAX_POSE_DIFF_DEG:
        score = 0.0

    return {
        "score": score,
        "base_similarity": base_similarity,
        "pose_diff": pose_diff,
        "yaw1": yaw1,
        "pitch1": pitch1,
        "yaw2": yaw2,
        "pitch2": pitch2,
    }


def verify_face(
    auth_image: np.ndarray, target_user: str | None = None
) -> tuple[str | None, float]:
    """
    Compare authentication image against registered models.
    Returns (matched_user_id, confidence) or (None, 0.0) if no match.
    """
    print("\nDebug Info:")

    result = get_face_features(auth_image)
    if result is None:
        print("- Failed to extract features from authentication image")
        return None, 0.0

    auth_features, auth_depth_variance = result
    print(f"- Auth depth variance: {auth_depth_variance:.6f}")

    if auth_depth_variance < MIN_DEPTH_VARIANCE:
        print(
            f"- LIVENESS CHECK FAILED: Depth variance {auth_depth_variance:.6f} below threshold"
        )
        print("- Possible 2D photo/screen detected (spoofing attempt)")
        return None, 0.0

    users = [target_user] if target_user else load_registered_models()
    if not users or users == [None]:
        return None, 0.0

    best_match = None
    best_score = 0.0

    print(f"- Found registered users: {users}")
    for user_id in users:
        model_img_path = os.path.join(
            "registered_faces", user_id, f"{user_id}_avgface.png"
        )
        if not os.path.exists(model_img_path):
            print(f"- Missing averaged face for user {user_id}: {model_img_path}")
            continue

        model_result = get_cached_model_features(user_id, model_img_path)
        if model_result is None:
            continue

        model_features, model_depth_variance = model_result
        print(f"- Model depth variance: {model_depth_variance:.6f}")
        print(f"- Auth features shape: {auth_features.shape}")
        print(f"- Model features shape: {model_features.shape}")

        metrics = compare_features(auth_features, model_features)
        score = metrics["score"]

        print(f"\n- Comparison for {user_id}:")
        print(
            f"  - Auth pose (yaw, pitch): ({metrics['yaw1']:.1f}°, {metrics['pitch1']:.1f}°)"
        )
        print(
            f"  - Model pose (yaw, pitch): ({metrics['yaw2']:.1f}°, {metrics['pitch2']:.1f}°)"
        )
        print(f"  - Pose difference: {metrics['pose_diff']:.1f}°")
        print(f"  - Base similarity: {metrics['base_similarity']:.4f}")
        print(f"  - Final confidence: {score:.1%}")

        if score > best_score:
            best_score = score
            best_match = user_id

    if best_score < MIN_CONFIDENCE:
        print(
            f"\n- Best score {best_score:.1%} below minimum threshold {MIN_CONFIDENCE:.1%}"
        )
        return None, 0.0

    return best_match, best_score
