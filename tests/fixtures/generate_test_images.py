#!/usr/bin/env python3
"""
Generate test images for PhotoChron tests.
"""

from pathlib import Path
import sys

# Add src to path to import photochron modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tests.fixtures.images import create_test_image, create_sample_photo_with_exif


def generate_sample_images() -> None:
    """Generate 5 sample test images with different characteristics."""
    output_dir = Path(__file__).parent.parent / "data" / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test images in {output_dir}")

    # Image 1: Basic test image
    img1 = create_test_image(width=800, height=600, color=(100, 150, 200))
    img1.rename(output_dir / "test_image_1.jpg")
    print(f"  Created: test_image_1.jpg")

    # Image 2: Portrait orientation
    img2 = create_test_image(width=600, height=800, color=(200, 100, 150))
    img2.rename(output_dir / "test_image_2.jpg")
    print(f"  Created: test_image_2.jpg")

    # Image 3: With EXIF metadata (recent date)
    img3 = create_sample_photo_with_exif(
        datetime_original="2023:07:15 14:30:00", make="TestCamera", model="Model X"
    )
    img3.rename(output_dir / "test_with_exif.jpg")
    print(f"  Created: test_with_exif.jpg")

    # Image 4: With faces (simulated)
    from tests.fixtures.images import create_face_test_image

    img4 = create_face_test_image(num_faces=2)
    img4.rename(output_dir / "test_with_faces.jpg")
    print(f"  Created: test_with_faces.jpg")

    # Image 5: Small image
    img5 = create_test_image(width=400, height=300, color=(50, 100, 50))
    img5.rename(output_dir / "test_small.jpg")
    print(f"  Created: test_small.jpg")

    print(f"\nGenerated {len(list(output_dir.glob('*.jpg')))} test images")


if __name__ == "__main__":
    generate_sample_images()
