"""OpenAI-compatible chat client. This module only calls the API."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_API_ATTEMPTS = 3
_RETRY_DELAYS = (0.5, 1.0, 2.0)
_RETRY_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

logger = logging.getLogger("pipeline")

T = TypeVar("T")


class LLMClientError(Exception):
    """Base error raised by the LLM client."""


class MissingAPIKeyError(LLMClientError):
    """OPENAI_API_KEY is not set."""


class EmptyModelResponseError(LLMClientError):
    """The model returned no text."""


class LLMAPIError(LLMClientError):
    """The LLM API rejected the request or was unreachable."""


class TransientLLMError(LLMClientError):
    """A temporary API failure that call_with_retry may try again."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def call_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = MAX_API_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run operation again after a temporary API failure or an empty response."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (TransientLLMError, EmptyModelResponseError) as exc:
            kind = getattr(exc, "kind", type(exc).__name__)
            logger.warning(
                "retry attempt=%s/%s error=%s: %s",
                attempt,
                attempts,
                kind,
                exc,
            )
            if attempt >= attempts:
                if isinstance(exc, EmptyModelResponseError):
                    raise
                raise LLMAPIError(str(exc)) from exc
            delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
            sleep(delay)

    raise LLMAPIError("LLM API error: no attempts were made")


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
        # The SDK retries on its own. call_with_retry is the only retry loop.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._sleep = time.sleep

    def complete(self, system: str, user: str) -> str:
        """Send a system and user message and return the raw model text."""
        return call_with_retry(
            lambda: self._complete_once(system, user),
            sleep=self._sleep,
        )

    def _complete_once(self, system: str, user: str) -> str:
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
        except APIConnectionError as exc:
            raise TransientLLMError(
                f"LLM API error: {exc}",
                kind=type(exc).__name__,
            ) from exc
        except APIStatusError as exc:
            if exc.status_code in _RETRY_STATUS_CODES:
                raise TransientLLMError(
                    f"LLM API error: {exc}",
                    kind=type(exc).__name__,
                ) from exc
            raise LLMAPIError(f"LLM API error: {exc}") from exc
        except APIError as exc:
            raise LLMAPIError(f"LLM API error: {exc}") from exc

        if not response.choices:
            raise EmptyModelResponseError("The model returned no choices.")

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise EmptyModelResponseError("The model returned an empty response.")
        return content
