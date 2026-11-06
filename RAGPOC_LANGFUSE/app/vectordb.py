"""ChromaDB storage and retrieval logic.

This module owns the persistent vector database used by the RAG pipeline.
It is responsible for storing embeddings and fetching the most relevant chunks
for a user question.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chromadb

from app.config import settings


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent Chroma client using the configured database path."""
    return chromadb.PersistentClient(path=str(settings.chroma_db_path))


def get_collection() -> Any:
    """Get or create the collection used for document embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=settings.collection_name)


def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """Insert chunk documents and vectors into ChromaDB.

    Args:
        chunks: Chunk dictionaries with text and metadata.
        embeddings: Embeddings corresponding to each chunk.
    """
    collection = get_collection()

    documents = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_similar_chunks(query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """Search the database for the most relevant chunks.

    Args:
        query_embedding: Embedding for the user's question.
        limit: Maximum number of results to return.

    Returns:
        List of dictionaries with document text, metadata, and similarity score.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[Dict[str, Any]] = []

    for index, document in enumerate(docs):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else 0.0
        similarity = max(0.0, 1.0 - float(distance))

        retrieved.append(
            {
                "text": document,
                "metadata": metadata,
                "score": similarity,
            }
        )

    return retrieved
