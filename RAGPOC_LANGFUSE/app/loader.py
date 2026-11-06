"""PDF loading and text extraction for the simple RAG pipeline.

This module is responsible for reading a PDF file from disk and converting its
pages into plain text. It intentionally does not perform chunking or embedding;
that separation keeps the system easy to understand and debug.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader


def load_pdf_text(pdf_path: str | Path) -> List[str]:
    """Read a PDF file and return plain text from each page.

    Args:
        pdf_path: File system path to a PDF document.

    Returns:
        A list of strings where each string represents one page of text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the provided path is not a PDF file.
        Exception: If the PDF cannot be read.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_file}")

    try:
        reader = PdfReader(str(pdf_file))
        pages: List[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            cleaned_text = " ".join(text.split())
            pages.append(cleaned_text)

        return pages
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF: {pdf_file}") from exc


# Example usage for learning purposes only.
if __name__ == "__main__":
    sample_path = "docs/hr_policy.pdf"
    try:
        pages = load_pdf_text(sample_path)
        print(f"Loaded {len(pages)} pages.")
        print(pages[0][:300] if pages else "No text extracted.")
    except Exception as exc:
        print(f"Error: {exc}")
