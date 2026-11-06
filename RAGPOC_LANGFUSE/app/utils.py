"""Small helper functions used across the project.

Keep utilities tiny and obvious. This file exists to avoid duplication of simple
operations such as sanitizing input or formatting output.
"""

from __future__ import annotations

from pathlib import Path


def normalize_pdf_path(value: str) -> Path:
    """Convert a user-supplied PDF path into a normalized Path object."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def format_source_line(filename: str, page_number: int) -> str:
    """Return a display-friendly source label for a retrieved chunk."""
    return f"{filename} | Page {page_number}"
