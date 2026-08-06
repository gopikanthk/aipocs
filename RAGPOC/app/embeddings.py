"""Embedding generation for chunks.

This module creates simple plain-Python embeddings for chunks so the RAG pipeline
can run without requiring an external embedding service.
"""

from __future__ import annotations

from typing import Any, List

from app.config import settings


def _get_local_sentence_model(model_name: str | None = None):
    """Load a SentenceTransformer model if it is available locally."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("sentence-transformers is not available")

    model = model_name or settings.embedding_model
    return SentenceTransformer(model)


def _fallback_embedding(text: str) -> List[float]:
    """Provide a lightweight local embedding when model downloads are blocked."""
    values: List[float] = []
    for index in range(384):
        char = text[index % len(text)] if text else " "
        value = ((ord(char) % 17) - 8) / 100.0
        if index % 2 == 0:
            value *= -1.0
        values.append(float(value))
    return values


def generate_embedding(text: str, model: str | None = None, force_local: bool = True) -> List[float]:
    """Create a local embedding for a single text string using SentenceTransformer when possible."""
    model_name = model or settings.embedding_model

    if not force_local:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.llm_base_url,
            )
            response = client.embeddings.create(
                model=model_name,
                input=text,
            )
            embedding = response.data[0].embedding
            return list(embedding)
        except Exception:
            pass

    try:
        model = _get_local_sentence_model(model_name)
        embedding = model.encode(text, convert_to_numpy=False)
        return [float(value) for value in embedding]
    except Exception:
        return _fallback_embedding(text)


def generate_embeddings_for_chunks(chunks: List[dict[str, Any]]) -> List[List[float]]:
    """Generate embeddings for every chunk in the list."""
    embeddings: List[List[float]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        embedding = generate_embedding(text)
        embeddings.append(embedding)

    return embeddings
