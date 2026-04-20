"""
EXIF field writer for Mode B output (enriched copies).

Writes DateTimeOriginal, ImageDescription and UserComment using piexif. Only
JPEG files are supported by piexif; non-JPEG extensions are copied verbatim
and a warning is logged.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger


def _format_datetime(year: int | None, month: int | None) -> str | None:
    if year is None:
        return None
    m = month if month else 1
    return f"{year:04d}:{m:02d}:01 00:00:00"


def _build_description(
    year: int | None,
    confidence: float,
    signals: dict[str, Any],
    review_needed: bool,
) -> str:
    year_str = str(year) if year is not None else "unknown"
    parts = [f"Est. {year_str}"]
    if "llm_decade" in signals and signals["llm_decade"].get("confidence") is not None:
        parts.append(f"LLM {signals['llm_decade'].get('confidence'):.2f}")
    if "face" in signals:
        parts.append("face age signal")
    if "photo_medium" in signals:
        parts.append("medium prior")
    parts.append(f"conf={confidence:.2f}")
    if review_needed:
        parts.append("review")
    return " - ".join(parts)


def write_exif_fields(
    target_path: Path,
    year: int | None,
    month: int | None,
    confidence: float,
    signals: dict[str, Any],
    review_needed: bool,
    full_result: dict[str, Any],
) -> bool:
    """Write PhotoChron EXIF fields to an existing JPEG file.

    Returns True if EXIF was written, False if skipped (e.g. unsupported format
    or piexif missing).
    """
    suffix = target_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        logger.debug("Skipping EXIF write for non-JPEG: {}", target_path)
        return False

    try:
        import piexif
    except ImportError:
        logger.warning("piexif not installed; cannot write EXIF to {}", target_path)
        return False

    try:
        try:
            exif_dict = piexif.load(str(target_path))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        exif_dict.setdefault("0th", {})
        exif_dict.setdefault("Exif", {})

        dt = _format_datetime(year, month)
        if dt:
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt.encode("ascii")

        description = _build_description(year, confidence, signals, review_needed)
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode(
            "utf-8", errors="replace"
        )

        user_comment = json.dumps(full_result, separators=(",", ":"))
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
            b"ASCII\x00\x00\x00" + user_comment.encode("utf-8", errors="replace")
        )

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(target_path))
        return True
    except Exception as exc:  # pragma: no cover - runtime safety net
        logger.warning("Failed to write EXIF for {}: {}", target_path, exc)
        return False
