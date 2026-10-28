"""Negative cases for retry, fallback, and length guardrails. No API key."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from openai import APIConnectionError, BadRequestError

from llm_client import LLMClient
from main import process_text, run_items
from prompts import ROUTE_ANSWER_LIMITS

TEXT = "Не могу войти в аккаунт, проверьте доступ."
EXTRACT = {
    "meaning": "Пользователь не может войти в аккаунт.",
    "key_points": ["ошибка входа", "нужна помощь", "аккаунт недоступен"],
}
CLASSIFY = {
    "category": "support",
    "sentiment": "neutral",
    "intent": "починить вход",
}
ANSWER = {"final_answer": "Проверьте пароль и сеть."}
CHECK = {"consistent": True, "details_kept": True, "notes": "детали на месте"}


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _chain() -> list[str]:
    return [_dump(EXTRACT), _dump(CLASSIFY), _dump(ANSWER), _dump(CHECK)]


class ScriptedClient:
    """Returns queued model texts. One bad text, then the repair, fits in the queue."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.model = "scripted"
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if not self.replies:
            raise AssertionError("scripted client ran out of replies")
        return self.replies.pop(0)


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def create(self, **kwargs: object) -> _Response:
        self.calls += 1
        if not self.actions:
            raise AssertionError("fake API ran out of actions")
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        if not isinstance(action, str):
            raise AssertionError(f"unexpected action {action!r}")
        return _Response(action)


def _api_client(actions: list[object]) -> tuple[LLMClient, _Completions]:
    completions = _Completions(actions)
    client = LLMClient.__new__(LLMClient)
    client.model = "fake"
    client._sleep = lambda _delay: None
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def _httpx():
    try:
        import httpx
    except ImportError:
        import httpx2 as httpx
    return httpx


def _connection_error() -> APIConnectionError:
    request = _httpx().Request("POST", "https://api.openai.com/v1/chat/completions")
    return APIConnectionError(message="temporary outage", request=request)


def _bad_request() -> BadRequestError:
    httpx = _httpx()
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("bad request", response=response, body=None)


def _warning_prefixes(warnings: list[str], prefix: str) -> bool:
    return any(item.startswith(prefix) for item in warnings)


def check_text_instead_of_json() -> None:
    client = ScriptedClient(
        ["Конечно, вот ответ обычным текстом без JSON.", *_chain()],
    )
    result = process_text(client, TEXT)
    assert not client.replies, "fallback reply was not used"
    assert result.summary == EXTRACT["meaning"]
    assert result.category.value == "support"
    assert _warning_prefixes(result.warnings, "fallback: extract_meaning:")
    assert not _warning_prefixes(result.warnings, "degraded:")


def check_missing_keys() -> None:
    client = ScriptedClient(
        [
            _dump(EXTRACT),
            _dump({"category": "support"}),
            _dump(CLASSIFY),
            _dump(ANSWER),
            _dump(CHECK),
        ],
    )
    result = process_text(client, TEXT)
    assert not client.replies, "repair reply was not used"
    assert result.category.value == "support"
    assert result.sentiment.value == "neutral"
    assert result.intent == CLASSIFY["intent"]
    assert _warning_prefixes(result.warnings, "fallback: classify:")
    assert not _warning_prefixes(result.warnings, "degraded:")


def check_api_retry() -> None:
    client, completions = _api_client(
        [_connection_error(), _connection_error(), *_chain()],
    )
    result = process_text(client, TEXT)
    assert completions.calls == 6, completions.calls
    assert not completions.actions
    assert result.category.value == "support"
    assert result.warnings == []


def check_empty_then_next() -> None:
    client, completions = _api_client(["", "", "", *_chain()])
    results = run_items(
        client,
        [
            {"id": "empty", "text": TEXT},
            {"id": "next", "text": "Сколько стоит тариф?"},
        ],
    )
    assert completions.calls == 7, completions.calls
    assert "error" in results[0]
    assert "пустой" in results[0]["error"]
    assert "category" not in results[0]
    assert "error" not in results[1]
    assert results[1]["category"] == "support"


def check_answer_too_long() -> None:
    limit = ROUTE_ANSWER_LIMITS["sales"]
    sales = {
        "category": "sales",
        "sentiment": "neutral",
        "intent": "узнать цену",
    }
    client = ScriptedClient(
        [
            _dump(EXTRACT),
            _dump(sales),
            _dump({"final_answer": "а" * 500}),
            _dump({"final_answer": "я" * 500}),
            _dump(CHECK),
        ],
    )
    result = process_text(client, "Сколько стоит тариф и чем он лучше?")
    assert not client.replies, "the long repair reply was not used"
    assert result.category.value == "sales"
    assert result.final_answer == "я" * limit
    assert len(result.final_answer) <= limit
    assert _warning_prefixes(result.warnings, "fallback: final_answer:")
    assert "truncated: final_answer" in result.warnings
    assert not _warning_prefixes(result.warnings, "degraded:")
    assert result.consistent is True


def check_late_steps_degrade() -> None:
    client = ScriptedClient(
        [
            _dump(EXTRACT),
            _dump(CLASSIFY),
            "не json",
            "снова не json",
            "и проверка не json",
            "проверка тоже не json",
        ],
    )
    result = process_text(client, TEXT)
    assert not client.replies, "degraded steps did not consume the bad replies"
    assert result.summary == EXTRACT["meaning"]
    assert result.final_answer == EXTRACT["meaning"]
    assert result.consistent is False
    assert result.details_kept is False
    assert result.check_notes.startswith("проверка не выполнена:")
    assert _warning_prefixes(result.warnings, "degraded: final_answer:")
    assert _warning_prefixes(result.warnings, "degraded: self_check:")


def check_bad_request_is_not_retried() -> None:
    client, completions = _api_client([_bad_request(), _dump(EXTRACT)])
    results = run_items(client, [{"id": "bad-request", "text": TEXT}])
    assert completions.calls == 1, completions.calls
    assert len(completions.actions) == 1
    assert "error" in results[0]
    assert "LLM API error" in results[0]["error"]


CHECKS = (
    ("текст вместо JSON", check_text_instead_of_json),
    ("часть ключей отсутствует", check_missing_keys),
    ("API временно недоступно", check_api_retry),
    ("пустой ответ, следующий текст жив", check_empty_then_next),
    ("слишком длинный ответ", check_answer_too_long),
    ("сбой ответа и проверки", check_late_steps_degrade),
    ("ошибка 400 без retry", check_bad_request_is_not_retried),
)


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
