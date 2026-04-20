"""
Output filename generation.
"""

import re
from pathlib import Path

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize(name: str) -> str:
    """Remove characters that are unsafe on common filesystems."""
    return _INVALID_CHARS.sub("_", name).strip()


def build_renamed_filename(
    sort_rank: int, estimated_year: int | None, original_name: str
) -> str:
    """Return '{rank:04d}_{year}-est_{original_name}' with safe characters.

    Unknown year becomes 'unknown'. The original extension is preserved.
    """
    rank = f"{sort_rank:04d}"
    year = str(estimated_year) if estimated_year is not None else "unknown"
    original = _sanitize(Path(original_name).name)
    return f"{rank}_{year}-est_{original}"
