"""
Ingestion stage: Read photos, compute hashes, downsample, extract EXIF.
"""

import hashlib
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import imagehash
import piexif
from loguru import logger

# Imported directly so the except clause resolves to a real class even when
# tests patch the `piexif` module reference wholesale.
from piexif import InvalidImageDataError
from PIL import Image

from photochron.config import get_config
from photochron.pipeline import PipelineStage, register_stage
from photochron.store import get_store


@register_stage
class IngestionStage(PipelineStage):
    """Stage 1: Photo ingestion and preprocessing."""

    def __init__(self):
        """Initialize ingestion stage with configuration."""
        self.config = get_config()
        self.supported_extensions = set(self.config.ingestion.supported_formats)
        self._hash_cache: dict[Path, str] = {}  # Cache of file path to perceptual hash

    @property
    def name(self) -> str:
        return "ingestion"

    @property
    def dependencies(self) -> list[str]:
        return []  # First stage, no dependencies

    def run(self, run_id: str, config_hash: str) -> None:
        """
        Process photos from input directory.

        For each photo:
        1. Scan input directory for image files
        2. Compute perceptual hash (pHash)
        3. Create downsampled version (max 1024px)
        4. Extract EXIF metadata (timestamp, camera, GPS)
        5. Store in photos table
        """
        logger.info(f"Starting ingestion stage for run {run_id}")

        if not self.config.input_dir:
            raise RuntimeError(
                "input_dir is not set – call PipelineRunner.run_pipeline() "
                "or assign config.input_dir before running this stage."
            )
        input_dir = Path(self.config.input_dir)
        cache_dir = Path(self.config.cache_dir)
        downsampled_dir = cache_dir / "downsampled"
        downsampled_dir.mkdir(parents=True, exist_ok=True)

        # Get list of image files
        image_files = self._scan_image_files(input_dir)
        total_files = len(image_files)

        if total_files == 0:
            logger.warning(f"No image files found in {input_dir}")
            self.mark_complete(run_id, photos_processed=0)
            return

        logger.info(f"Found {total_files} image files to process")

        processed_count = 0
        for i, file_path in enumerate(image_files):
            try:
                self._process_image(file_path, downsampled_dir, run_id)
                processed_count += 1

                # Report progress every 10 files or at the end
                if (i + 1) % 10 == 0 or (i + 1) == total_files:
                    logger.info(f"Processed {i + 1}/{total_files} files")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Continue with next file

        logger.info(f"Ingestion complete. Processed {processed_count}/{total_files} files successfully")
        self.mark_complete(run_id, photos_processed=processed_count)

    def _scan_image_files(self, input_dir: Path) -> list[Path]:
        """Scan directory for image files with supported extensions."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        # Remove duplicates (case-insensitive)
        seen = set()
        unique_files = []
        for file in image_files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)

        return sorted(unique_files)

    def _process_image(self, file_path: Path, downsampled_dir: Path, run_id: str) -> None:
        """Process a single image file."""
        try:
            # Compute content hash first (for duplicate detection)
            content_hash = self._compute_content_hash(file_path)

            # Check if photo with this content hash already exists
            existing_photo = self._get_existing_photo(content_hash)
            cached_perceptual_hash = existing_photo.get("perceptual_hash") if existing_photo else None

            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Compute perceptual hash if not cached
                if cached_perceptual_hash:
                    phash_hex = cached_perceptual_hash
                    logger.debug(f"Using cached perceptual hash for {file_path}")
                else:
                    phash = imagehash.phash(img)
                    phash_hex = str(phash)

                # Get basic metadata
                width, height = img.size
                format_name = img.format or "UNKNOWN"

                # Create downsampled version (skip if already exists and cached)
                downsampled_path = None
                if not cached_perceptual_hash or not existing_photo.get("downsample_path"):
                    downsampled_path = self._create_downsampled_image(img, phash_hex, downsampled_dir, format_name)
                else:
                    # Use existing downsampled path
                    downsampled_path_str = existing_photo.get("downsample_path")
                    downsampled_path = Path(downsampled_path_str) if downsampled_path_str else None

                # Extract EXIF metadata (always extract, in case metadata changed)
                exif_data = self._extract_exif_metadata(file_path)

                # Store in database (INSERT OR REPLACE will update if content_hash exists)
                self._store_photo_metadata(
                    content_hash=content_hash,
                    file_path=str(file_path),
                    downsampled_path=str(downsampled_path) if downsampled_path else None,
                    perceptual_hash=phash_hex,
                    width=width,
                    height=height,
                    format_name=format_name,
                    exif_datetime=exif_data.get("datetime"),
                    make=exif_data.get("make"),
                    model=exif_data.get("model"),
                    run_id=run_id,
                )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

    def _create_downsampled_image(
        self,
        img: Image.Image,
        phash_hex: str,
        downsampled_dir: Path,
        original_format: str,
    ) -> Path | None:
        """Create downsampled version of image (max size from config)."""
        max_size = self.config.ingestion.max_downsample_size
        width, height = img.size

        # If image is already smaller than max_size, don't create new file
        if max(width, height) <= max_size:
            return None

        # Calculate new dimensions preserving aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Determine output format (use JPEG for photos, PNG for transparency)
        output_format = "JPEG" if original_format in ("JPEG", "JPG", "HEIC", "RAW") else "PNG"
        output_ext = ".jpg" if output_format == "JPEG" else ".png"

        # Create filename from perceptual hash
        output_path = downsampled_dir / f"{phash_hex}{output_ext}"

        # Save with appropriate quality
        save_kwargs = {}
        if output_format == "JPEG":
            save_kwargs["quality"] = 85
            save_kwargs["optimize"] = True

        resized_img.save(output_path, format=output_format, **save_kwargs)
        return output_path

    def _extract_exif_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract EXIF metadata from image file."""
        exif_data = {}

        try:
            # First try with piexif for detailed EXIF
            exif_dict = piexif.load(str(file_path))

            # Extract DateTimeOriginal
            if "Exif" in exif_dict and piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
                dt_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
                # Convert to ISO 8601 format
                try:
                    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                    exif_data["datetime"] = dt.isoformat()
                except ValueError:
                    exif_data["datetime"] = dt_str

            # Extract camera make and model
            if "0th" in exif_dict:
                if piexif.ImageIFD.Make in exif_dict["0th"]:
                    exif_data["make"] = exif_dict["0th"][piexif.ImageIFD.Make].decode("utf-8").strip()
                if piexif.ImageIFD.Model in exif_dict["0th"]:
                    exif_data["model"] = exif_dict["0th"][piexif.ImageIFD.Model].decode("utf-8").strip()

            # Extract GPS coordinates if available and enabled in config
            if self.config.ingestion.extract_gps and "GPS" in exif_dict:
                gps_data = exif_dict["GPS"]
                lat, lon = self._parse_gps_coordinates(gps_data)
                if lat is not None and lon is not None:
                    exif_data["gps_latitude"] = lat
                    exif_data["gps_longitude"] = lon

        except (InvalidImageDataError, ValueError, KeyError, TypeError, struct.error, OSError) as e:
            # Expected piexif failure modes (malformed EXIF, missing GPS segment,
            # struct decoding errors). Fall back to Pillow's loose EXIF parsing.
            logger.debug(f"piexif failed for {file_path}, trying Pillow: {e}")
            try:
                with Image.open(file_path) as img:
                    exif = img.getexif()
                    if exif:
                        # Get DateTimeOriginal (tag 36867)
                        dt = exif.get(36867)
                        if dt:
                            exif_data["datetime"] = dt

                        # Get Make (tag 271) and Model (tag 272)
                        make = exif.get(271)
                        if make:
                            exif_data["make"] = str(make).strip()
                        model = exif.get(272)
                        if model:
                            exif_data["model"] = str(model).strip()
            except Exception as e2:
                logger.debug(f"Pillow EXIF also failed: {e2}")

        # If no EXIF datetime, use file modification time
        if "datetime" not in exif_data:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            exif_data["datetime"] = mtime.isoformat()
            exif_data["datetime_source"] = "file_mtime"

        return exif_data

    def _parse_gps_coordinates(self, gps_data: dict) -> tuple[float | None, float | None]:
        """Parse GPS coordinates from EXIF GPS dictionary."""
        try:
            # Parse latitude
            if piexif.GPSIFD.GPSLatitude in gps_data and piexif.GPSIFD.GPSLatitudeRef in gps_data:
                lat = self._convert_gps_coordinate(
                    gps_data[piexif.GPSIFD.GPSLatitude],
                    gps_data[piexif.GPSIFD.GPSLatitudeRef],
                )
            else:
                lat = None

            # Parse longitude
            if piexif.GPSIFD.GPSLongitude in gps_data and piexif.GPSIFD.GPSLongitudeRef in gps_data:
                lon = self._convert_gps_coordinate(
                    gps_data[piexif.GPSIFD.GPSLongitude],
                    gps_data[piexif.GPSIFD.GPSLongitudeRef],
                )
            else:
                lon = None

            return lat, lon
        except Exception:
            return None, None

    def _convert_gps_coordinate(self, coord: tuple, ref: bytes) -> float:
        """Convert GPS coordinate from (degrees, minutes, seconds) to decimal degrees."""
        degrees, minutes, seconds = coord
        degrees = degrees[0] / degrees[1] if isinstance(degrees, tuple) else degrees
        minutes = minutes[0] / minutes[1] if isinstance(minutes, tuple) else minutes
        seconds = seconds[0] / seconds[1] if isinstance(seconds, tuple) else seconds

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        # Apply direction reference
        if ref in [b"S", b"W"]:
            decimal = -decimal

        return decimal

    def _compute_content_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for duplicate detection."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _get_existing_photo(self, content_hash: str) -> dict[str, Any] | None:
        """Check if a photo with this content hash already exists in the database."""
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT perceptual_hash, file_path, downsample_path, exif_datetime, make, model
                FROM photos 
                WHERE content_hash = ?
                LIMIT 1
                """,
                (content_hash,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "perceptual_hash": row[0],
                    "file_path": row[1],
                    "downsample_path": row[2],
                    "exif_datetime": row[3],
                    "make": row[4],
                    "model": row[5],
                }
        return None

    def _store_photo_metadata(
        self,
        content_hash: str,
        file_path: str,
        downsampled_path: str | None,
        perceptual_hash: str,
        width: int,
        height: int,
        format_name: str,
        exif_datetime: str | None,
        make: str | None,
        model: str | None,
        run_id: str,
    ) -> None:
        """Store photo metadata in the database."""
        store = get_store()
        with store.transaction() as conn:
            # Use INSERT OR REPLACE to handle duplicates
            conn.execute(
                """
                INSERT OR REPLACE INTO photos 
                (content_hash, file_path, downsample_path, exif_datetime, make, model, perceptual_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    file_path,
                    downsampled_path,
                    exif_datetime,
                    make,
                    model,
                    perceptual_hash,
                ),
            )

            # Also record that this photo was processed in this run
            # (This could be expanded to a separate photo_runs table in the future)
            logger.debug(f"Stored photo: {file_path} (hash: {content_hash[:8]})")
