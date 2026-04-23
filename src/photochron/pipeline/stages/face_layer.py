"""
Face layer stage: Detect faces, compute embeddings, estimate ages.
"""

import sqlite3
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from photochron.config import get_config
from photochron.face.insightface_wrapper import InsightFaceWrapper
from photochron.models import FaceCreate
from photochron.pipeline import PipelineStage, register_stage
from photochron.store import get_store


@register_stage
class FaceLayerStage(PipelineStage):
    """Stage 2: Face detection and analysis."""

    def __init__(self) -> None:
        """Initialize face layer stage with configuration."""
        self.config = get_config()
        self.face_config = self.config.face
        self.wrapper = InsightFaceWrapper(
            model_name=self.face_config.model_name,
            detection_threshold=self.face_config.detection_threshold,
            use_gpu=self.face_config.use_gpu,
        )

    @property
    def name(self) -> str:
        return "face_layer"

    @property
    def dependencies(self) -> list[str]:
        return ["ingestion"]  # Depends on photos being ingested

    def run(self, run_id: str, config_hash: str) -> None:
        """
        Detect faces and compute age estimates.

        For each photo without face data:
        1. Load downsampled image
        2. Run InsightFace detection
        3. Compute embeddings and age estimates
        4. Match to known persons or mark as unknown
        5. Store in faces table with confidence scores
        """
        logger.info("Starting face layer stage")
        try:
            photos = self._get_photos_without_faces()
            if not photos:
                logger.info("No photos without face data; stage complete")
                self.mark_complete(run_id, photos_processed=0)
                return

            total_photos = len(photos)
            logger.info(f"Found {total_photos} photos without face data")

            # Load model (lazy load via wrapper)
            # Trigger loading by calling detect_faces on a dummy image
            # We'll just let it load on first detection

            processed = 0
            batch_size = self.face_config.batch_size
            if batch_size > 1:
                logger.debug(f"Using batch size {batch_size} (sequential processing)")

            for i in range(0, total_photos, batch_size):
                batch = photos[i : i + batch_size]
                for row in batch:
                    photo_id, downsample_path = row["id"], Path(row["downsample_path"])
                    try:
                        self._process_photo(photo_id, downsample_path)
                        processed += 1
                        if processed % 10 == 0:
                            logger.info(f"Processed {processed}/{total_photos} photos")
                    except (OSError, ValueError, RuntimeError, Image.UnidentifiedImageError) as e:
                        # Per-photo failures (unreadable file, decoder error, ONNX
                        # runtime error) must not abort the whole batch; anything
                        # else indicates a bug and should propagate.
                        logger.warning(f"Failed to process photo {photo_id}: {e}")
                        continue

            logger.info(f"Face layer stage completed. Processed {processed}/{total_photos} photos")
            self.mark_complete(run_id, photos_processed=processed)

        except (OSError, RuntimeError, sqlite3.DatabaseError) as e:
            logger.error(f"Face layer stage failed: {e}")
            raise

    def _get_photos_without_faces(self) -> list:
        """Query photos that have no face records yet."""
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT p.id, p.downsample_path
                FROM photos p
                LEFT JOIN faces f ON p.id = f.photo_id
                WHERE f.id IS NULL
                ORDER BY p.id
                """
            )
            return cursor.fetchall()

    def _load_downsampled_image(self, downsample_path: Path) -> np.ndarray:
        """
        Load downsampled image from disk as RGB numpy array.

        Args:
            downsample_path: Path to downsampled image (from photos table)

        Returns:
            RGB image array (height, width, 3) with values 0-255, uint8
        """
        if not downsample_path.exists():
            raise FileNotFoundError(f"Downsampled image not found: {downsample_path}")
        with Image.open(downsample_path) as img:
            img = img.convert("RGB")
            return np.array(img)

    def _crop_face_with_margin(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        margin_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Crop face from image with optional margin.

        Args:
            image: RGB image array (height, width, 3)
            bbox: Bounding box (x1, y1, x2, y2) in absolute pixel coordinates
            margin_ratio: Additional margin as ratio of bounding box dimensions

        Returns:
            Cropped face image (RGB, uint8)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)

        # Expand bounding box with margin, clamp to image boundaries
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box after margin: ({x1}, {y1}, {x2}, {y2})")

        return image[y1:y2, x1:x2]

    def _person_embedding_column_exists(self) -> bool:
        """Check if persons table has an embedding column."""
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute("PRAGMA table_info(persons)")
            columns = [row[1] for row in cursor.fetchall()]
            return "embedding" in columns

    def _get_known_persons_with_embeddings(self) -> list[dict]:
        """
        Retrieve known persons with their embeddings (if available).

        Returns:
            List of dicts with keys: id, person_id, name, birthday, embedding (np.ndarray or None)
        """
        store = get_store()
        persons = []
        with store.transaction() as conn:
            # Check if embedding column exists
            cursor = conn.execute("PRAGMA table_info(persons)")
            columns = [row[1] for row in cursor.fetchall()]
            has_embedding = "embedding" in columns

            if has_embedding:
                cursor = conn.execute("SELECT id, person_id, name, birthday, embedding FROM persons")
                for row in cursor.fetchall():
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
                    persons.append(
                        {
                            "id": row["id"],
                            "person_id": row["person_id"],
                            "name": row["name"],
                            "birthday": row["birthday"],
                            "embedding": embedding,
                        }
                    )
            else:
                cursor = conn.execute("SELECT id, person_id, name, birthday FROM persons")
                for row in cursor.fetchall():
                    persons.append(
                        {
                            "id": row["id"],
                            "person_id": row["person_id"],
                            "name": row["name"],
                            "birthday": row["birthday"],
                            "embedding": None,
                        }
                    )
        return persons

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _match_person(self, embedding: np.ndarray) -> int | None:
        """
        Match embedding against known persons.

        Returns:
            person_id if match above threshold, else None.
        """
        persons = self._get_known_persons_with_embeddings()
        if not persons:
            return None

        best_similarity = -1.0
        best_person_id = None
        for person in persons:
            if person["embedding"] is None:
                continue
            similarity = self._cosine_similarity(embedding, person["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person["id"]

        if best_similarity >= self.face_config.matching_threshold:
            return best_person_id
        return None

    def _process_photo(self, photo_id: int, downsample_path: Path) -> None:
        """Process a single photo: detect faces, compute embeddings, estimate ages."""
        logger.debug(f"Processing photo {photo_id}: {downsample_path}")
        try:
            image = self._load_downsampled_image(downsample_path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping photo {photo_id}: {e}")
            return

        # Detect faces
        detections = self.wrapper.detect_faces(image)
        if not detections:
            logger.debug(f"No faces detected in photo {photo_id}")
            return

        faces = []
        for bbox, confidence in detections:
            try:
                # Crop face with margin
                cropped = self._crop_face_with_margin(image, bbox)
                # Compute embedding
                embedding = self.wrapper.compute_embedding(cropped)
                # Estimate age
                age_mean, age_std = self.wrapper.estimate_age(cropped)
                # Scale standard deviation by configurable factor
                age_std *= self.face_config.age_confidence_scale
                # Ensure minimum std dev of 1 year
                age_std = max(age_std, 1.0)

                # Match person
                person_id = self._match_person(embedding)

                face_record = {
                    "photo_id": photo_id,
                    "person_id": person_id,
                    "embedding": embedding,
                    "age_estimate": age_mean,
                    "age_std": age_std,
                    "confidence": confidence,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                }
                faces.append(face_record)
                logger.debug(
                    f"Detected face in photo {photo_id}: confidence {confidence:.3f}, age {age_mean:.1f}±{age_std:.1f}"
                )
            except ValueError as e:
                logger.warning(f"Failed to process face in photo {photo_id}: {e}")
                continue

        if faces:
            self._store_faces(photo_id, faces)
            logger.info(f"Stored {len(faces)} faces for photo {photo_id}")

    def _store_faces(self, photo_id: int, faces: list[dict]) -> None:
        """Store face detection results in database."""
        if not faces:
            return

        store = get_store()
        try:
            with store.transaction() as conn:
                # Delete existing faces for this photo (simplifies upsert)
                conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))

                # Insert new face records
                for face in faces:
                    # Convert embedding numpy array to bytes
                    embedding_bytes = face["embedding"].tobytes() if face["embedding"] is not None else None

                    face_record = FaceCreate(
                        photo_id=face["photo_id"],
                        person_id=face["person_id"],
                        embedding=embedding_bytes,
                        age_estimate=face["age_estimate"],
                        age_std=face["age_std"],
                        confidence=face["confidence"],
                        bbox_x1=face["bbox_x1"],
                        bbox_y1=face["bbox_y1"],
                        bbox_x2=face["bbox_x2"],
                        bbox_y2=face["bbox_y2"],
                    )
                    # Insert record
                    conn.execute(
                        """
                        INSERT INTO faces
                        (photo_id, person_id, embedding, age_estimate, age_std, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            face_record.photo_id,
                            face_record.person_id,
                            face_record.embedding,
                            face_record.age_estimate,
                            face_record.age_std,
                            face_record.confidence,
                            face_record.bbox_x1,
                            face_record.bbox_y1,
                            face_record.bbox_x2,
                            face_record.bbox_y2,
                        ),
                    )
            logger.debug(f"Stored {len(faces)} faces for photo {photo_id}")
        except sqlite3.Error as e:
            logger.error(f"Database error storing faces for photo {photo_id}: {e}")
            raise
