"""Ingestion script for building the ChromaDB document store.

This script should be run when the user wants to index a new PDF document.
It reads the file, chunks the text, creates embeddings, and stores them in the
persistent vector database.
"""

from __future__ import annotations

from app.chunker import chunk_pages
from app.config import settings
from app.embeddings import generate_embeddings_for_chunks
from app.loader import load_pdf_text
from app.vectordb import store_chunks


def ingest_pdf(pdf_path: str) -> None:
    """Run the full ingestion pipeline for a single PDF file."""
    print("Loading PDF...")
    pages = load_pdf_text(pdf_path)

    print("Extracting text...")
    print("Chunking...")
    chunks = chunk_pages(pages, filename=pdf_path, chunk_size=1000, chunk_overlap=200)

    print("Generating embeddings...")
    embeddings = generate_embeddings_for_chunks(chunks)

    print("Saving vectors...")
    store_chunks(chunks, embeddings)

    print(f"Completed. Indexed {len(chunks)} chunks from {pdf_path}.")


if __name__ == "__main__":
    try:
        pdf_path = input("Enter PDF path: ").strip()
        ingest_pdf(pdf_path)
    except Exception as exc:
        print(f"Error: {exc}")
