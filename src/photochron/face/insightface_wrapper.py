"""
InsightFace wrapper for face detection, embedding extraction, and age estimation.

Encapsulates the InsightFace model and provides a clean interface for the face layer.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from loguru import logger


class InsightFaceWrapper:
    """Wrapper around InsightFace for face detection and analysis."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        """
        Initialize the InsightFace wrapper.

        Args:
            model_name: Name of InsightFace model (e.g., "buffalo_l", "buffalo_s")
            detection_threshold: Minimum confidence for face detection (0.0-1.0)
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.model_name = model_name
        self.detection_threshold = detection_threshold
        self.use_gpu = use_gpu
        self._model: Optional[Any] = None
        self._providers = ["CPUExecutionProvider"]
        if use_gpu:
            # Try CUDA provider if GPU is requested
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def load_model(self) -> None:
        """Load InsightFace model (lazy initialization)."""
        if self._model is not None:
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "InsightFace not installed. Please install via 'pip install insightface'"
            ) from e

        logger.info(f"Loading InsightFace model '{self.model_name}'")
        self._model = FaceAnalysis(name=self.model_name, providers=self._providers)
        ctx_id = 0 if self.use_gpu else -1  # -1 for CPU, 0 for GPU device 0
        self._model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace model loaded successfully")

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in an image.

        Args:
            image: RGB image as numpy array (height, width, 3), uint8

        Returns:
            List of tuples (bbox, confidence) where bbox is (x1, y1, x2, y2)
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None  # for type checker

        # Run detection
        detections = self._model.get(image)

        results = []
        for det in detections:
            bbox = det.bbox.astype(int)
            confidence = det.det_score
            if confidence >= self.detection_threshold:
                results.append(((bbox[0], bbox[1], bbox[2], bbox[3]), confidence))

        logger.debug(
            f"Detected {len(results)} faces (threshold={self.detection_threshold})"
        )
        return results

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Compute 512‑dimensional face embedding.

        Args:
            face_image: Cropped face image (RGB, uint8)

        Returns:
            Normalized embedding vector (512‑dim)
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None  # for type checker

        # The embedding is already computed during detection and stored in det.embedding
        # This method is for when we have a cropped face image separately.
        # For simplicity, we'll reuse the detection pipeline on the cropped image.
        # Note: This is suboptimal; a better approach would extract embedding directly.
        # For now, we'll run detection on the cropped image (should find exactly one face).
        detections = self._model.get(face_image)
        if len(detections) == 0:
            raise ValueError("No face found in the cropped image")
        # Use the first detection's embedding
        embedding = detections[0].embedding
        # Normalize to unit length (already normalized by InsightFace?)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def estimate_age(self, face_image: np.ndarray) -> Tuple[float, float]:
        """
        Estimate age of a face.

        Args:
            face_image: Cropped face image (RGB, uint8)

        Returns:
            Tuple (mean_age, std_dev) where std_dev is scaled by configurable factor
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None  # for type checker

        # InsightFace's buffalo_l model includes gender/age estimation.
        # The age is stored in det.age.
        detections = self._model.get(face_image)
        if len(detections) == 0:
            raise ValueError("No face found in the cropped image")
        age = detections[0].age
        # Standard deviation is not provided by InsightFace; we'll use a configurable scale.
        # For now, return a fixed relative uncertainty (e.g., 10% of age).
        # This will be scaled by age_confidence_scale in the caller.
        std_dev = max(age * 0.1, 1.0)  # at least 1 year
        return float(age), float(std_dev)

    def batch_detect(
        self, images: List[np.ndarray]
    ) -> List[List[Tuple[Tuple[int, int, int, int], float]]]:
        """
        Detect faces in a batch of images.

        Args:
            images: List of RGB images (same dimensions)

        Returns:
            List of detection results per image
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None  # for type checker

        # InsightFace does not support batch inference out of the box.
        # We'll simply loop; for GPU acceleration, consider custom batching.
        results = []
        for img in images:
            results.append(self.detect_faces(img))
        return results

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
