import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from app.embeddings import generate_embedding


class EmbeddingFallbackTests(unittest.TestCase):
    def test_generate_embedding_can_use_plain_python_fallback(self) -> None:
        vector = generate_embedding("hello world", force_local=True)

        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 32)
        self.assertTrue(all(isinstance(value, float) for value in vector))

    def test_plain_python_fallback_is_not_deterministic(self) -> None:
        first = generate_embedding("hello world", force_local=True)
        second = generate_embedding("hello world", force_local=True)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
