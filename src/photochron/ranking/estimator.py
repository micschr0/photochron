"""
Weighted date-estimate combination for the ranking engine.

Combines four signals per photo:
  1. face_age - estimated photo year from (person.birthday + face.age_estimate)
  2. llm_decade - midpoint year of decade bucket returned by the vision LLM
  3. photo_medium_prior - era prior based on physical medium (print, polaroid, digital)
  4. exif_date - deterministic override when EXIF DateTimeOriginal is present

Weights come from ConfigPipeline. EXIF, when present, replaces the weighted
combine entirely with confidence 1.0.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

MEDIUM_PRIORS: dict[str, tuple[int, float]] = {
    # medium label -> (year_midpoint, confidence)
    "digital": (2010, 0.6),
    "print_scan": (1985, 0.35),
    "print": (1985, 0.35),
    "polaroid": (1978, 0.45),
    "slide": (1970, 0.4),
    "film": (1975, 0.35),
    "black_and_white": (1955, 0.4),
    "sepia": (1920, 0.4),
    "unknown": (1990, 0.2),
}


@dataclass
class DateEstimate:
    """Aggregated year estimate with breakdown of contributing signals."""

    year: int | None = None
    month: int | None = None
    confidence: float = 0.0
    signals: dict[str, dict[str, float]] = field(default_factory=dict)
    review_needed: bool = False
    notes: str = ""


def decade_midpoint(decade: str | None) -> int | None:
    """Return the midpoint year of a 'YYYY-YYYY' decade string, or None."""
    if not decade:
        return None
    try:
        a, b = decade.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        return (start + end) // 2
    except (ValueError, AttributeError):
        return None


def face_year_estimate(
    taken_on_or_before: date,
    birthday: str | None,
    age_estimate: float | None,
) -> int | None:
    """Return year = birthday.year + age_estimate, if both are available."""
    if not birthday or age_estimate is None:
        return None
    try:
        bd = date.fromisoformat(birthday)
    except ValueError:
        return None
    year = bd.year + int(round(age_estimate))
    current_year = taken_on_or_before.year
    if year > current_year:
        year = current_year
    return year


def medium_prior_year(medium: str | None) -> tuple[int, float] | None:
    """Look up (year, confidence) prior for a photo medium label."""
    if not medium:
        return None
    return MEDIUM_PRIORS.get(medium.lower())


def _parse_exif_year(exif_datetime: str | None) -> tuple[int | None, int | None]:
    """Parse EXIF DateTimeOriginal → (year, month) with tolerant formats."""
    if not exif_datetime:
        return None, None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y:%m:%d"):
        try:
            parsed = datetime.strptime(exif_datetime, fmt)
            return parsed.year, parsed.month
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(exif_datetime)
        return parsed.year, parsed.month
    except ValueError:
        return None, None


def combine_signals(
    *,
    exif_datetime: str | None,
    face_year: int | None,
    face_confidence: float | None,
    decade: str | None,
    decade_confidence: float | None,
    photo_medium: str | None,
    photo_medium_confidence: float | None,
    weights: dict[str, float],
    min_confidence_threshold: float,
) -> DateEstimate:
    """Combine available signals into a single DateEstimate.

    `weights` contains keys 'face', 'llm', 'medium'.
    EXIF overrides everything when present.
    """
    estimate = DateEstimate()

    exif_year, exif_month = _parse_exif_year(exif_datetime)
    if exif_year is not None:
        estimate.year = exif_year
        estimate.month = exif_month
        estimate.confidence = 1.0
        estimate.signals["exif"] = {"year": float(exif_year), "confidence": 1.0}
        return estimate

    llm_year = decade_midpoint(decade)
    medium_entry = medium_prior_year(photo_medium)

    numerator = 0.0
    denom = 0.0
    signals: dict[str, dict[str, float]] = {}

    if face_year is not None:
        w = weights.get("face", 0.0) * max(face_confidence or 0.0, 0.0)
        if w > 0:
            numerator += face_year * w
            denom += w
            signals["face"] = {
                "year": float(face_year),
                "confidence": float(face_confidence or 0.0),
                "weight": float(w),
            }

    if llm_year is not None:
        w = weights.get("llm", 0.0) * max(decade_confidence or 0.0, 0.0)
        if w > 0:
            numerator += llm_year * w
            denom += w
            signals["llm_decade"] = {
                "year": float(llm_year),
                "confidence": float(decade_confidence or 0.0),
                "weight": float(w),
            }

    if medium_entry is not None:
        medium_year, medium_base_conf = medium_entry
        effective_conf = medium_base_conf
        if photo_medium_confidence is not None:
            effective_conf *= photo_medium_confidence
        w = weights.get("medium", 0.0) * max(effective_conf, 0.0)
        if w > 0:
            numerator += medium_year * w
            denom += w
            signals["photo_medium"] = {
                "year": float(medium_year),
                "confidence": float(effective_conf),
                "weight": float(w),
            }

    if denom == 0:
        estimate.signals = signals
        estimate.review_needed = True
        estimate.notes = "No usable signals"
        return estimate

    weighted_year = numerator / denom

    total_weight_max = (
        weights.get("face", 0.0) + weights.get("llm", 0.0) + weights.get("medium", 0.0)
    )
    normalized = denom / total_weight_max if total_weight_max > 0 else 0.0
    confidence = max(0.0, min(1.0, normalized))

    estimate.year = int(round(weighted_year))
    estimate.confidence = confidence
    estimate.signals = signals
    estimate.review_needed = confidence < min_confidence_threshold
    return estimate


def rank_estimates(
    photo_estimates: list[tuple[int, DateEstimate]],
) -> list[tuple[int, int]]:
    """Return [(photo_id, sort_rank)] sorted by (year, month, photo_id).

    Photos with missing year are placed at the end, preserving insertion order.
    """
    def key(item: tuple[int, DateEstimate]) -> tuple[int, int, int, int]:
        photo_id, est = item
        has_year = 0 if est.year is not None else 1
        return (
            has_year,
            est.year if est.year is not None else 0,
            est.month if est.month is not None else 0,
            photo_id,
        )

    ordered = sorted(photo_estimates, key=key)
    return [(photo_id, rank) for rank, (photo_id, _) in enumerate(ordered)]
