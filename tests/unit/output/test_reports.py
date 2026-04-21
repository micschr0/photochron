"""Tests for output report and timeline builders."""

from photochron.output import build_report, build_timeline_rows


def _rows():
    return [
        {
            "photo_id": 1,
            "sort_rank": 0,
            "estimated_year": 1985,
            "estimated_month": 7,
            "confidence": 0.8,
            "review_needed": False,
            "original_name": "a.jpg",
            "output_renamed": "/out/renamed/0000_1985-est_a.jpg",
            "output_enriched": "/out/exif_enriched/a.jpg",
        },
        {
            "photo_id": 2,
            "sort_rank": 1,
            "estimated_year": None,
            "estimated_month": None,
            "confidence": 0.2,
            "review_needed": True,
            "original_name": "b.jpg",
            "output_renamed": "/out/renamed/0001_unknown-est_b.jpg",
            "output_enriched": "/out/exif_enriched/b.jpg",
        },
    ]


def test_build_report_summary_counts_correctly():
    report = build_report("run_abc", _rows())
    summary = report["summary"]
    assert summary["total_photos"] == 2
    assert summary["photos_with_year"] == 1
    assert summary["review_needed"] == 1
    assert summary["year_range"] == [1985, 1985]
    assert report["run_id"] == "run_abc"


def test_build_report_empty_has_null_year_range():
    report = build_report("run_x", [])
    assert report["summary"]["year_range"] is None
    assert report["summary"]["total_photos"] == 0


def test_build_timeline_csv_has_header_and_rows():
    csv_text = build_timeline_rows(_rows())
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("sort_rank,estimated_year,")
    assert len(lines) == 3  # header + 2 rows
    assert "a.jpg" in lines[1]
    assert "b.jpg" in lines[2]


def test_build_timeline_csv_sorts_by_rank():
    rows = _rows()[::-1]  # reversed
    csv_text = build_timeline_rows(rows)
    lines = csv_text.strip().splitlines()
    assert lines[1].startswith("0,")
    assert lines[2].startswith("1,")
