"""
face_model.py — embedding via InsightFace Python package

This module wraps InsightFace's FaceAnalysis to produce L2-normalized
512-D embeddings from face crops (BGR, e.g., 224x224 from your pipeline).

Usage:
    from face_model import FaceEmbedder
    import cv2

    emb = FaceEmbedder(model_pack="buffalo_l")
    img = cv2.imread("registered_faces/test_user/sample.jpg")
    vec = emb.embed(img)
    print(vec.shape, vec[:5])

Dependencies (install):
    pip install insightface onnxruntime-silicon opencv-python numpy

Notes:
- First run will auto-download the model pack to ~/.insightface/models
- Default provider is CPUExecutionProvider; you can add CoreML if available.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2

try:
    # InsightFace API
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover
    raise ImportError(
        "insightface is not installed. Run: pip install insightface onnxruntime-silicon"
    ) from e


class FaceEmbedder:
    """High-level wrapper around InsightFace's FaceAnalysis.

    Parameters
    ----------
    model_pack : str
        Model pack name from InsightFace model zoo. Recommended: "buffalo_l".
    providers : Optional[List[str]]
        ONNX Runtime providers. Defaults to ["CPUExecutionProvider"].
        You may try ["CoreMLExecutionProvider", "CPUExecutionProvider"] if available.
    det_size : Tuple[int, int]
        Detection input size (w, h). Smaller can speed up, 128–256 is fine for crops.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (192, 192),
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(name=model_pack, providers=providers)
        # ctx_id=0 selects CPU when providers includes CPUExecutionProvider
        self._app.prepare(ctx_id=0, det_size=det_size)

    @staticmethod
    def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected BGR HxWx3 image")
        return img

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            return v
        return v / n

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Compute a single L2-normalized embedding from a face crop (BGR).

        If multiple faces are detected in the crop, the largest face is used.
        Returns a (512,) float32 vector.
        """
        img = self._ensure_bgr_uint8(face_bgr)
        faces = self._app.get(img)
        if not faces:
            raise ValueError("No face detected in the provided image/crop.")

        # Choose the largest face by bounding-box area if multiple are present
        if len(faces) > 1:
            areas = [float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) for f in faces]
            face = faces[int(np.argmax(areas))]
        else:
            face = faces[0]

        emb = face.normed_embedding.astype(np.float32)
        # Safety re-normalization (InsightFace already provides normed vectors)
        emb = self._l2_normalize(emb)
        return emb

    def embed_many(self, crops_bgr: Iterable[np.ndarray]) -> np.ndarray:
        """Batch-embed a list/iterable of BGR crops -> (N, 512) float32 array.
        Skips crops with no detectable face.
        """
        out: List[np.ndarray] = []
        for img in crops_bgr:
            try:
                out.append(self.embed(img))
            except Exception:
                # Skip images without a face; you can log if needed
                continue
        if not out:
            raise ValueError("No valid faces found in any of the provided crops.")
        return np.stack(out, axis=0)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        if b.dtype != np.float32:
            b = b.astype(np.float32)
        # If not normalized, normalize defensively
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test FaceEmbedder using InsightFace")
    parser.add_argument("image", type=str, help="Path to a face crop (BGR) image")
    parser.add_argument("--model_pack", type=str, default="buffalo_l", help="InsightFace model pack name")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    fe = FaceEmbedder(model_pack=args.model_pack)
    vec = fe.embed(img)
    print("Embedding shape:", vec.shape)
    print("First 8 values:", np.round(vec[:8], 5))