"""Interactive chat script for Q&A over a PDF knowledge base.

This script embeds the user's question, retrieves similar chunks from ChromaDB,
constructs a prompt, calls OpenRouter, and displays the answer with source
metadata and similarity scores.
"""

from __future__ import annotations

from app.config import settings
from app.embeddings import generate_embedding
from app.llm import generate_answer
from app.prompts import build_prompt
from app.vectordb import search_similar_chunks


def ask_question(question: str) -> None:
    """Run the retrieval and generation flow for a single question."""
    print("Loading vector database...")
    question_embedding = generate_embedding(question)

    print("Searching...")
    results = search_similar_chunks(question_embedding, limit=settings.top_k)

    if not results:
        print("No relevant information found in the uploaded document.")
        return

    context = "\n\n".join(
        f"Source: {item['metadata'].get('filename', 'unknown')} | "
        f"Page {item['metadata'].get('page_number', 'unknown')}\n"
        f"{item['text']}"
        for item in results
    )

    print("Calling OpenRouter...")
    prompt = build_prompt(context=context, question=question)
    answer = generate_answer(prompt)

    print("\nAnswer")
    print(answer)
    print("\nSources")
    for item in results:
        metadata = item["metadata"]
        print(f"{metadata.get('filename', 'unknown')}")
        print(f"Page {metadata.get('page_number', 'unknown')}")
        print(f"Similarity {item['score']:.2f}")


if __name__ == "__main__":
    try:
        question = input("Enter your question: ").strip()
        ask_question(question)
    except Exception as exc:
        print(f"Error: {exc}")
