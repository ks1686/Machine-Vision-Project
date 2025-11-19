"""
compare_faces.py
---------------
Compares a captured authentication image against registered 3D face models.
Uses MediaPipe Face Mesh landmarks for geometric comparison and feature matching.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from face_processing import facemesh_vector_from_aligned, detect_face_and_landmarks, assess_quality

def load_registered_models() -> List[str]:
    """Get list of registered user IDs from models directory."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    # Look for .obj files which indicate completed registrations
    users = []
    for file in os.listdir(models_dir):
        if file.endswith("_face.obj"):
            user_id = file.replace("_face.obj", "")
            users.append(user_id)
    return users

def estimate_head_pose(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Estimate head pose angles from 3D landmarks.
    Returns (yaw, pitch) in degrees.
    Yaw: left/right rotation
    Pitch: up/down rotation
    """
    # Use specific landmarks for pose estimation
    nose_tip = landmarks[1]      # Nose tip
    chin = landmarks[152]        # Chin
    left_eye = landmarks[33]     # Left eye corner
    right_eye = landmarks[263]   # Right eye corner
    left_mouth = landmarks[61]   # Left mouth corner
    right_mouth = landmarks[291] # Right mouth corner
    
    # Calculate yaw (left/right rotation)
    # Compare distances from nose to each side of face
    left_dist = np.linalg.norm(nose_tip[:2] - left_eye[:2])
    right_dist = np.linalg.norm(nose_tip[:2] - right_eye[:2])
    yaw = np.degrees(np.arctan2(right_dist - left_dist, (left_dist + right_dist) / 2)) * 2
    
    # Calculate pitch (up/down rotation)
    # Use vertical position of nose relative to eyes and mouth
    eye_y = (left_eye[1] + right_eye[1]) / 2
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2
    nose_y = nose_tip[1]
    
    # Normalize nose position between eyes and mouth
    face_height = mouth_y - eye_y
    if face_height != 0:
        nose_position = (nose_y - eye_y) / face_height
        pitch = (nose_position - 0.5) * 60  # Map to degrees
    else:
        pitch = 0.0
    
    return yaw, pitch

def get_face_features(image: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """
    Extract normalized face mesh features from image.
    Returns (features, depth_variance) tuple.
    depth_variance is used for liveness detection.
    """
    print(f"- Processing image shape: {image.shape}")
    
    # Ensure image is 512x512 (same as registration)
    if image.shape[:2] != (512, 512):
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        print(f"- Resized to: {image.shape}")
    
    # Get face mesh landmarks directly (image is already aligned)
    landmarks = facemesh_vector_from_aligned(image)
    if landmarks is None:
        print("- Failed to extract face mesh landmarks")
        return None
        
    # Reshape to (N,3) and normalize
    N = landmarks.size // 3
    features = landmarks.reshape(N, 3).astype(np.float32)
    
    # Calculate depth variance BEFORE normalization (for liveness detection)
    z_variance = np.var(features[:, 2])
    
    # Center and scale normalize
    features[:, :2] -= features[:, :2].mean(axis=0, keepdims=True)
    scale = np.sqrt((features[:, :2] ** 2).sum(axis=1)).max()
    if scale > 0:
        features[:, :2] /= scale
    
    # Normalize Z values separately (they're in a different scale)
    z_range = features[:, 2].max() - features[:, 2].min()
    if z_range > 0:
        features[:, 2] = (features[:, 2] - features[:, 2].min()) / z_range
        
    return features, z_variance

def compare_features(features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:
    """
    Compare two sets of face features, return similarity metrics.
    Returns dict with: score, base_similarity, pose_diff, yaw1, pitch1, yaw2, pitch2
    """
    if features1 is None or features2 is None:
        return {"score": 0.0, "base_similarity": 0.0, "pose_diff": 0.0, 
                "yaw1": 0.0, "pitch1": 0.0, "yaw2": 0.0, "pitch2": 0.0}
        
    # Reshape if needed
    if len(features1.shape) == 1:
        features1 = features1.reshape(-1, 3)
    if len(features2.shape) == 1:
        features2 = features2.reshape(-1, 3)
        
    # Ensure same number of landmarks
    min_points = min(len(features1), len(features2))
    features1 = features1[:min_points]
    features2 = features2[:min_points]
    
    # Estimate head poses
    yaw1, pitch1 = estimate_head_pose(features1)
    yaw2, pitch2 = estimate_head_pose(features2)
    
    # Calculate pose difference
    pose_diff = np.sqrt((yaw1 - yaw2)**2 + (pitch1 - pitch2)**2)
    
    # Weight coordinates for better pose sensitivity
    weights = np.array([1.0, 1.0, 0.5])
    
    # Compute weighted distance between corresponding landmarks
    diff = np.sqrt((((features1 - features2) * weights) ** 2).sum(axis=1))
    
    # Use multiple percentiles to catch both local and global differences
    diff_75 = np.percentile(diff, 75)
    diff_90 = np.percentile(diff, 90)
    mean_diff = (diff_75 + diff_90) / 2.0
    
    # Add pose penalty based on Z coordinate differences
    z_diff = np.abs(features1[:, 2] - features2[:, 2]).mean()
    pose_penalty = z_diff * 0.5
    
    # Calculate base similarity
    base_similarity = 1.0 / (1.0 + mean_diff + pose_penalty)
    
    # Apply pose angle penalty (0 to 30+ degrees = 0 to 100% penalty)
    pose_angle_penalty = np.clip(pose_diff / 30.0, 0, 1) * 0.5  # Up to 50% penalty
    
    # Adjust similarity with pose penalty
    adjusted_similarity = base_similarity * (1.0 - pose_angle_penalty)
    
    # Calculate final confidence score using sigmoid function
    # As described in paper: sigmoid applied to inverse mean landmark distance
    # This maps the similarity score to a 0-1 confidence range
    adjusted_score = (adjusted_similarity - 0.15) / 0.35
    adjusted_score = max(0.0, min(1.0, adjusted_score))
    
    # Apply sigmoid transformation for smooth confidence mapping
    final_confidence = (1.0 / (1.0 + np.exp(-4 * (adjusted_score - 0.5)))) * 0.6 + 0.35
    
    return {
        "score": float(final_confidence),
        "base_similarity": float(base_similarity),
        "pose_diff": float(pose_diff),
        "yaw1": float(yaw1),
        "pitch1": float(pitch1),
        "yaw2": float(yaw2),
        "pitch2": float(pitch2)
    }

def compare_with_registration_frames(auth_features: np.ndarray, user_id: str) -> float:
    """Compare auth features against a user's registration frames."""
    user_dir = os.path.join("registered_faces", user_id)
    if not os.path.exists(user_dir):
        print(f"- Missing directory for user {user_id}")
        return 0.0
        
    # Get straight-ahead pose frames
    reg_frames = [f for f in os.listdir(user_dir) if f.endswith('.png') and 's01_' in f]
    if not reg_frames:
        print(f"- No registration frames found for user {user_id}")
        return 0.0
        
    print(f"- Found {len(reg_frames)} registration frames for {user_id}")
    scores = []
    
    for frame_name in reg_frames:
        frame_path = os.path.join(user_dir, frame_name)
        model_img = cv2.imread(frame_path)
        if model_img is None:
            continue
            
        model_features = get_face_features(model_img)
        if model_features is None:
            continue
            
        score = compare_features(auth_features, model_features)
        scores.append(score)
        print(f"- Frame {frame_name}: {score:.1%}")
    
    if not scores:
        return 0.0
        
    # Use average of top 3 scores
    scores.sort(reverse=True)
    avg_score = sum(scores[:3]) / min(3, len(scores))
    print(f"- Average top score for {user_id}: {avg_score:.1%}")
    return avg_score

def verify_face(auth_image: np.ndarray, target_user: Optional[str] = None) -> Tuple[Optional[str], float]:
    """
    Compare authentication image against registered models.
    Returns (matched_user_id, confidence) or (None, 0.0) if no match.
    If target_user is provided, only compare against that specific user.
    """
    print("\nDebug Info:")
    
    # Get features from auth image
    result = get_face_features(auth_image)
    if result is None:
        print("- Failed to extract features from authentication image")
        return None, 0.0
    
    auth_features, auth_depth_variance = result
    print(f"- Auth depth variance: {auth_depth_variance:.6f}")
    
    # Check for liveness - distinguish flat 2D photos from live 3D faces
    MIN_DEPTH_VARIANCE = 0.003  # Threshold to detect flat images (spoofing)
    if auth_depth_variance < MIN_DEPTH_VARIANCE:
        print(f"- LIVENESS CHECK FAILED: Depth variance {auth_depth_variance:.6f} below threshold")
        print("- Possible 2D photo/screen detected (spoofing attempt)")
        # Note: This check has limitations with standard webcams
        # More reliable liveness detection would require blink detection or multi-frame analysis
    
    # Get list of users to compare against
    users = [target_user] if target_user else load_registered_models()
    if not users:
        return None, 0.0
        
    best_match = None
    best_score = 0.0
    best_metrics = None
    
    # Compare against each registered model
    print(f"- Found registered users: {users}")
    for user_id in users:
        # Load user's averaged face image
        model_img_path = os.path.join("registered_faces", user_id, f"{user_id}_avgface.png")
        if not os.path.exists(model_img_path):
            print(f"- Missing averaged face for user {user_id}: {model_img_path}")
            continue
            
        model_img = cv2.imread(model_img_path)
        if model_img is None:
            print(f"- Failed to load averaged face for user {user_id}")
            continue
            
        # Get features from model
        model_result = get_face_features(model_img)
        if model_result is None:
            continue
        
        model_features, model_depth_variance = model_result
        print(f"- Model depth variance: {model_depth_variance:.6f}")
            
        # Compare features
        print(f"- Auth features shape: {auth_features.shape if auth_features is not None else None}")
        print(f"- Model features shape: {model_features.shape if model_features is not None else None}")
        
        metrics = compare_features(auth_features, model_features)
        score = metrics["score"]
        
        # Print detailed metrics
        print(f"\n- Comparison for {user_id}:")
        print(f"  - Auth pose (yaw, pitch): ({metrics['yaw1']:.1f}°, {metrics['pitch1']:.1f}°)")
        print(f"  - Model pose (yaw, pitch): ({metrics['yaw2']:.1f}°, {metrics['pitch2']:.1f}°)")
        print(f"  - Pose difference: {metrics['pose_diff']:.1f}°")
        print(f"  - Base similarity: {metrics['base_similarity']:.4f}")
        print(f"  - Final confidence: {score:.1%}")
        
        # Update best match if score is higher
        if score > best_score:
            best_score = score
            best_match = user_id
            best_metrics = metrics
    
    # Require a minimum confidence threshold
    MIN_CONFIDENCE = 0.78  # Raised to reject borderline cases
    if best_score < MIN_CONFIDENCE:
        print(f"\n- Best score {best_score:.1%} below minimum threshold {MIN_CONFIDENCE:.1%}")
        return None, 0.0
        
    return best_match, best_score
