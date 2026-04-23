#!/usr/bin/env python3
"""
Generate a synthetic PhotoChron benchmark fixture.

The generated photos are deliberately *not* real family photos – we want
repeatable, privacy-free benchmarking. Each image has:

* Random procedural content (gradient + noise + geometric shapes) so the
  JPEG encoder sees a realistic entropy profile instead of compressing a
  solid colour down to a few hundred bytes.
* Plausible pixel dimensions drawn from a mix of typical scan/phone sizes
  (see ``SIZE_MIX``). This exercises the Pillow resize path in the same
  shape range the real ingestion stage sees.
* Basic EXIF (``DateTimeOriginal``, ``Make``, ``Model``) on ~50% of files
  so the piexif code path gets covered; the rest go through the Pillow
  fallback.
* No GPS, no faces – the face layer will run but detect nothing, which
  is fine for measuring wrapper overhead and the Ollama context layer.

Usage::

    # 500 photos into ./bench_fixture/
    python scripts/gen_bench_fixture.py --count 500 --output ./bench_fixture

    # Reproducible fixture
    python scripts/gen_bench_fixture.py --count 1000 --seed 42
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import piexif
from PIL import Image, ImageDraw, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Iterable


# Mix of sizes loosely representative of digitised family photos + modern
# phone captures. Weights bias towards the realistic middle range.
SIZE_MIX: list[tuple[tuple[int, int], int]] = [
    ((800, 600), 2),
    ((1024, 768), 3),
    ((1600, 1200), 4),
    ((2048, 1536), 3),
    ((3024, 4032), 2),  # phone portrait
    ((4032, 3024), 2),  # phone landscape
]

FAKE_CAMERAS: list[tuple[str, str]] = [
    ("Canon", "PowerShot SX540 HS"),
    ("NIKON", "COOLPIX P7100"),
    ("Apple", "iPhone 12"),
    ("EPSON", "Perfection V600"),
    ("Sony", "DSC-HX300"),
]


def _weighted_size(rng: random.Random) -> tuple[int, int]:
    """Pick a size from ``SIZE_MIX`` according to its weight column."""
    sizes, weights = zip(*SIZE_MIX, strict=True)
    return rng.choices(sizes, weights=weights, k=1)[0]


def _paint_content(img: Image.Image, rng: random.Random) -> None:
    """Fill ``img`` with a gradient plus a few random shapes.

    This gives the JPEG encoder enough variation to produce a realistic
    file size (~200 KB–1 MB depending on dimensions) and makes imagehash
    produce stable, non-degenerate hashes instead of all-zeros.
    """
    width, height = img.size
    draw = ImageDraw.Draw(img)

    base_hue = rng.random()
    for y in range(0, height, 8):
        r = int(128 + 127 * ((y / height) - 0.5) * 2)
        g = int(128 + 127 * (base_hue - 0.5) * 2)
        b = int(128 - 127 * ((y / height) - 0.5) * 2)
        draw.rectangle((0, y, width, y + 8), fill=(r % 256, g % 256, b % 256))

    for _ in range(rng.randint(5, 15)):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        x1, y1 = x0 + rng.randint(20, 300), y0 + rng.randint(20, 300)
        colour = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() < 0.5:
            draw.ellipse((x0, y0, x1, y1), fill=colour)
        else:
            draw.rectangle((x0, y0, x1, y1), fill=colour)

    # Mild blur, so phash doesn't see pure DCT-friendly edges. Note:
    # ``ImageFilter`` is applied eagerly – no getdata flush needed.
    if rng.random() < 0.4:
        img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))


def _make_exif(rng: random.Random, base_date: datetime) -> bytes | None:
    """Return an EXIF bytestring for ~50% of files; None for the rest.

    The ``base_date`` is randomised forward by up to ~3 years so sorting
    algorithms later have something meaningful to order by.
    """
    if rng.random() >= 0.5:
        return None
    delta_days = rng.randint(0, 365 * 3)
    dt = base_date + timedelta(days=delta_days)
    make, model = rng.choice(FAKE_CAMERAS)
    exif_dict: dict[str, dict[int, object]] = {
        "0th": {
            piexif.ImageIFD.Make: make.encode("ascii"),
            piexif.ImageIFD.Model: model.encode("ascii"),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii"),
        },
    }
    try:
        return piexif.dump(exif_dict)
    except (piexif.InvalidImageDataError, ValueError, KeyError):
        return None


def _save_jpeg(img: Image.Image, path: Path, exif: bytes | None) -> None:
    buf = BytesIO()
    kwargs: dict[str, object] = {"quality": 88, "optimize": False}
    if exif is not None:
        kwargs["exif"] = exif
    img.save(buf, format="JPEG", **kwargs)
    path.write_bytes(buf.getvalue())


def generate_fixture(
    count: int,
    output: Path,
    seed: int | None = None,
) -> Iterable[Path]:
    """Generate ``count`` synthetic JPEGs into ``output``.

    Yields the generated paths as they are written so the caller can stream
    progress updates.
    """
    rng = random.Random(seed)
    base_date = datetime(2020, 1, 1, 12, 0, 0)
    output.mkdir(parents=True, exist_ok=True)
    digits = max(4, len(str(count)))
    for i in range(count):
        size = _weighted_size(rng)
        img = Image.new("RGB", size, (200, 200, 200))
        _paint_content(img, rng)
        exif = _make_exif(rng, base_date)
        path = output / f"synth_{i:0{digits}d}.jpg"
        _save_jpeg(img, path, exif)
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of photos to generate (default: 100)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./bench_fixture"),
        help="Output directory (default: ./bench_fixture)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible fixtures",
    )
    args = parser.parse_args()

    written = 0
    for path in generate_fixture(args.count, args.output, seed=args.seed):
        written += 1
        if written % 50 == 0 or written == args.count:
            print(f"  {written}/{args.count} ({path.name})")
    print(f"Generated {written} synthetic photos in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
