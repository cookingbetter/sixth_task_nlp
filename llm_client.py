"""OpenAI-compatible chat client. This module only calls the API."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import APIError, OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLMClientError(Exception):
    """Base error raised by the LLM client."""


class MissingAPIKeyError(LLMClientError):
    """OPENAI_API_KEY is not set."""


class EmptyModelResponseError(LLMClientError):
    """The model returned no text."""


class LLMAPIError(LLMClientError):
    """The LLM API rejected the request or was unreachable."""


class LLMClient:
    """Thin wrapper around a chat completion call."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add the key."
            )

        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        self.model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str) -> str:
        """Send a system and user message and return the raw model text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except APIError as exc:
            raise LLMAPIError(f"LLM API error: {exc}") from exc

        if not response.choices:
            raise EmptyModelResponseError("The model returned no choices.")

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise EmptyModelResponseError("The model returned an empty response.")
        return content
