"""
Mock image generation utilities for tests.
"""

import tempfile
from pathlib import Path

import piexif
from PIL import Image, ImageDraw


def create_test_image(
    width: int = 800,
    height: int = 600,
    color: tuple[int, int, int] = (100, 150, 200),
    format: str = "JPEG",
    exif: dict | None = None,
) -> Path:
    """
    Create a test image with optional EXIF metadata.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple
        format: Image format ('JPEG', 'PNG')
        exif: Optional EXIF metadata dict

    Returns:
        Path to temporary image file
    """
    # Create image
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # Add some content so image isn't uniform
    draw.rectangle([50, 50, width - 50, height - 50], outline=(200, 100, 50), width=5)
    draw.ellipse([100, 100, 200, 200], fill=(255, 255, 0))

    # Save to temporary file
    suffix = ".jpg" if format == "JPEG" else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        img_path = Path(f.name)

        if format == "JPEG" and exif:
            # Add EXIF metadata
            exif_bytes = piexif.dump(exif)
            img.save(img_path, format=format, exif=exif_bytes)
        else:
            img.save(img_path, format=format)

    return img_path


def create_sample_photo_with_exif(
    datetime_original: str | None = None,
    make: str = "Test Camera",
    model: str = "Test Model",
) -> Path:
    """
    Create a sample photo with EXIF metadata.

    Args:
        datetime_original: DateTimeOriginal in format 'YYYY:MM:DD HH:MM:SS'
        make: Camera make
        model: Camera model

    Returns:
        Path to temporary image file
    """
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: make.encode("utf-8"),
            piexif.ImageIFD.Model: model.encode("utf-8"),
        },
        "Exif": {},
    }

    if datetime_original:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime_original.encode("utf-8")

    return create_test_image(exif=exif_dict)


def create_face_test_image(
    num_faces: int = 1,
    face_positions: list | None = None,
) -> Path:
    """
    Create a test image with simulated faces.

    Args:
        num_faces: Number of faces to simulate
        face_positions: List of (x1, y1, x2, y2) tuples for face positions

    Returns:
        Path to temporary image file
    """
    width, height = 800, 600
    img = Image.new("RGB", (width, height), (150, 200, 150))
    draw = ImageDraw.Draw(img)

    # Default face positions if not provided
    if face_positions is None:
        face_positions = []
        for i in range(num_faces):
            x = 100 + i * 150
            face_positions.append((x, 100, x + 100, 200))

    # Draw faces as circles
    for x1, y1, x2, y2 in face_positions:
        # Face circle
        draw.ellipse([x1, y1, x2, y2], fill=(255, 220, 180))
        # Eyes
        draw.ellipse([x1 + 20, y1 + 30, x1 + 40, y1 + 50], fill=(0, 0, 0))
        draw.ellipse([x2 - 40, y1 + 30, x2 - 20, y1 + 50], fill=(0, 0, 0))
        # Mouth
        draw.arc([x1 + 25, y1 + 50, x2 - 25, y2 - 20], 0, 180, fill=(0, 0, 0), width=3)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = Path(f.name)
        img.save(img_path, format="JPEG")

    return img_path


def cleanup_test_images(image_paths):
    """
    Clean up temporary test images.

    Args:
        image_paths: List of Path objects to delete
    """
    for path in image_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
