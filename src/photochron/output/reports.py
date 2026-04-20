"""
JSON report and CSV timeline builders.
"""

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any


def build_report(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the photochron_report.json payload."""
    total = len(rows)
    review = sum(1 for r in rows if r.get("review_needed"))
    with_year = sum(1 for r in rows if r.get("estimated_year") is not None)
    avg_conf = (
        sum(float(r.get("confidence") or 0.0) for r in rows) / total if total else 0.0
    )

    years_present = [r["estimated_year"] for r in rows if r.get("estimated_year")]
    year_range = [min(years_present), max(years_present)] if years_present else None

    return {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_photos": total,
            "photos_with_year": with_year,
            "average_confidence": round(avg_conf, 4),
            "review_needed": review,
            "year_range": year_range,
        },
        "photos": rows,
    }


def build_timeline_rows(rows: list[dict[str, Any]]) -> str:
    """Return CSV content (string) for photochron_timeline.csv."""
    fieldnames = [
        "sort_rank",
        "estimated_year",
        "estimated_month",
        "confidence",
        "review_needed",
        "original_name",
        "output_renamed",
        "output_enriched",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r.get("sort_rank", 0)):
        writer.writerow({k: _stringify(row.get(k)) for k in fieldnames})
    return buf.getvalue()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def dump_report(path: str, run_id: str, rows: list[dict[str, Any]]) -> None:
    """Convenience: write the JSON report to disk."""
    payload = build_report(run_id, rows)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
