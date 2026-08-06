"""LLM interaction through a generic OpenAI-compatible connector.

This module is responsible for sending the final prompt to the configured model
and returning the answer text to the chat flow.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from app.config import settings


class LLMResponse(BaseModel):
    """A small schema for validating the model's answer."""

    answer: str = Field(min_length=1)


class AgentResult(BaseModel):
    """A small structured output model for agent-style interaction."""

    answer: str = Field(min_length=1)
    source: str = Field(default="unknown")


def get_llm_client() -> OpenAI:
    """Return the OpenAI-compatible client configured for the selected backend."""
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def generate_answer(prompt: str) -> str:
    """Send a prompt to the configured model and return the answer."""
    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        return "I could not find this information in the uploaded document."

    try:
        parsed = LLMResponse(answer=content.strip())
    except ValidationError:
        return "I could not find this information in the uploaded document."

    return parsed.answer


def generate_structured_answer(prompt: str) -> AgentResult:
    """Return a structured agent-style result using Pydantic validation."""
    answer_text = generate_answer(prompt)
    return AgentResult(answer=answer_text, source="llm")
