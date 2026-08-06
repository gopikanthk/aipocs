"""Shared data structures for the RAG pipeline.

These simple models help keep the code readable and reduce the risk of using
loose dictionaries throughout many modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    """A single chunk of document text with metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the chunk into a dictionary compatible with ChromaDB."""
        return {
            "id": self.chunk_id or self.metadata.get("chunk_id", ""),
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """A retrieved chunk with similarity information."""

    text: str
    metadata: Dict[str, Any]
    score: float
