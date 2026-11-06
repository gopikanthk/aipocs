"""Prompt construction for question answering.

This module keeps the system prompt separate from the business logic. That makes
it easier to read, change, and reason about how the model is instructed.
"""

from __future__ import annotations


def build_prompt(context: str, question: str) -> str:
    """Create a prompt that tells the model to answer only from the supplied context."""
    template = """You are an enterprise document assistant.

Answer ONLY using the supplied context.

If the answer cannot be found in the supplied context, respond with:

\"I could not find this information in the uploaded document.\"

Never hallucinate.

Always mention the source page.

Context
{context}

Question
{question}
"""

    return template.format(context=context, question=question)
