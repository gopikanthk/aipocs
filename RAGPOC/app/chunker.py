"""Text chunking for the simple RAG pipeline.

This module takes extracted PDF text and splits it into smaller pieces so they
can be embedded and stored in a vector database. It uses a recursive
character-based splitter and attaches metadata useful for retrieval and citation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[str],
    filename: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Split page text into chunks and attach metadata.

    Args:
        pages: List of page text strings.
        filename: Name of the source PDF file.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dictionaries with text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    file_name = str(Path(filename).name)

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{file_name}-p{page_number}-c{chunk_index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text.strip(),
                    "metadata": {
                        "filename": file_name,
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                    },
                }
            )

    return chunks


# Example usage for learning purposes only.
if __name__ == "__main__":
    sample_pages = [
        "This is page one with a lot of text. " * 30,
        "This is page two with some additional content. " * 20,
    ]

    result = chunk_pages(sample_pages, "sample.pdf")
    print(f"Created {len(result)} chunks.")
    print(result[0])
