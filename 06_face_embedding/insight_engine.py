"""High-Precision Custom Biometric Face Embedding Engine.

Directly powers facial identity representation using our custom-built
Inception-ResNet-v1 (512-D) architecture trained on VGGFace2.
Operates on pure TensorFlow/Keras and OpenCV without black-box library wrappers.
"""

import hashlib
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure local imports find custom_inception_resnet
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from custom_inception_resnet import load_or_build_custom_model


class MultimodalCustomCNNEngine:
    """Production 512-D Custom Inception-ResNet-v1 Biometric Feature Extractor."""

    def __init__(self, model_path: Optional[str] = None):
        self.embedding_dim = 512
        print(f"[INFO] Booting Custom Biometric Engine: Inception-ResNet-v1 ({self.embedding_dim}-D)...")

        # Load or compile the custom Keras architecture
        self.model = load_or_build_custom_model(model_path=model_path)

        # OpenCV Haar cascade detector for rapid face localization
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Warm-up pass to ensure GPU/CPU kernel readiness
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        self._extract_from_160(dummy)
        print(
            f"[SUCCESS] Custom Inception-ResNet-v1 Engine initialized and warmed up "
            f"({self.embedding_dim}-D embeddings ready)."
        )

    def _extract_from_160(self, img_160_rgb: np.ndarray) -> np.ndarray:
        """Runs the custom model on a 160x160 RGB face crop with standard FaceNet standardization."""
        img_f = img_160_rgb.astype(np.float32)
        mean, std = img_f.mean(), img_f.std()
        std_adj = np.maximum(std, 1.0 / np.sqrt(img_f.size))
        normalized = (img_f - mean) / std_adj
        batch = np.expand_dims(normalized, axis=0)

        raw_vec = self.model(batch, training=False).numpy().flatten()
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = raw_vec / norm
        return raw_vec

    def extract_norm_embedding_from_frame(
        self, frame_bgr: np.ndarray, use_tta: bool = False
    ) -> Optional[np.ndarray]:
        """Extracts 512-D L2-normalized embedding from either an aligned face crop

        or an uncropped camera frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]

        # If already an aligned face crop from SmartVisionEngine (close to square and under ~320px)
        if w <= 320 and h <= 320:
            face_crop = frame_bgr
        else:
            # Full camera frame: locate the face using Haar cascade
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces[0]
                pad_x, pad_y = int(fw * 0.1), int(fh * 0.1)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
                face_crop = frame_bgr[y1:y2, x1:x2]
            else:
                face_crop = frame_bgr

        # Resize to 160x160 and convert to RGB
        resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        vec = self._extract_from_160(rgb)

        if use_tta:
            flipped_rgb = cv2.flip(rgb, 1)
            vec_flip = self._extract_from_160(flipped_rgb)
            combined = vec + vec_flip
            c_norm = np.linalg.norm(combined)
            if c_norm > 0:
                vec = combined / c_norm

        return vec

    def generate_normalized_embedding(self, image_path: str) -> Tuple[np.ndarray, str]:
        """Reads image from disk, detects the face, extracts 512-D L2-normalized embedding,

        and returns the vector and its SHA-256 integrity hash.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to decode image at {image_path}")

        vector = self.extract_norm_embedding_from_frame(img)
        if vector is None:
            raise ValueError(f"Failed to generate embedding for {image_path}")

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        v_hash = hashlib.sha256(vector.tobytes()).hexdigest()
        return vector, v_hash

    def extract_batch_embeddings(
        self, frames_bgr: List[np.ndarray], use_tta: bool = False
    ) -> List[Optional[np.ndarray]]:
        """Batch feature extraction helper."""
        return [self.extract_norm_embedding_from_frame(f, use_tta=use_tta) for f in frames_bgr]

    def predict_liveness(self, frame_bgr: np.ndarray) -> float:
        """Anti-spoofing score stub [0.0 = Spoof, 1.0 = Real Face]."""
        return 1.0


# Legacy alias
CustomCNNEngine = MultimodalCustomCNNEngine
