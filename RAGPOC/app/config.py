"""Application configuration for the simple RAG project.

This module centralizes environment-based settings used across the pipeline.
It keeps configuration readable and easy to override without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Strongly typed runtime settings for the project."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str
    embedding_api_key: str
    embedding_model: str
    chroma_db_path: Path
    collection_name: str
    top_k: int


def get_settings() -> Settings:
    """Load settings from environment variables and validate essential values."""
    import os

    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant").strip()
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    chroma_db_path = Path(os.getenv("CHROMA_DB_PATH", "vector_db")).resolve()
    collection_name = os.getenv("COLLECTION_NAME", "documents").strip()

    top_k_raw = os.getenv("TOP_K", "3").strip()
    try:
        top_k = int(top_k_raw)
    except ValueError as exc:
        raise ValueError("TOP_K must be an integer value.") from exc

    if not llm_api_key:
        raise ValueError("LLM_API_KEY is required.")
    if not embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY is required.")
    if not collection_name:
        raise ValueError("COLLECTION_NAME cannot be empty.")
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    return Settings(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        top_k=top_k,
    )


settings = get_settings()
