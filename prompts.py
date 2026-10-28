"""Prompt variants for the text processing step."""

from collections.abc import Callable
from dataclasses import dataclass

MAX_SUMMARY_CHARS = 280
MAX_KEY_POINT_CHARS = 120
MAX_FINAL_ANSWER_CHARS = 200
KEY_POINT_COUNT = 3

DEFAULT_VARIANT_NAME = "plain"

_JSON_SHAPE = """\
{
  "summary": "краткое содержание",
  "category": "question | information | advice | bug | other",
  "sentiment": "positive | neutral | negative",
  "key_points": ["мысль 1", "мысль 2", "мысль 3"],
  "final_answer": "короткий полезный ответ или следующий шаг"
}"""


@dataclass(frozen=True)
class PromptVariant:
    """System prompt and user-message template for one formulation."""

    name: str
    system: str
    build_user: Callable[[str], str]


def _plain_user(text: str) -> str:
    return f"Текст:\n{text.strip()}"


def _checklist_user(text: str) -> str:
    return (
        "Ограничения:\n"
        f"- summary не длиннее {MAX_SUMMARY_CHARS} символов\n"
        f"- key_points: ровно {KEY_POINT_COUNT} пункта, каждый не длиннее {MAX_KEY_POINT_CHARS} символов\n"
        f"- final_answer не длиннее {MAX_FINAL_ANSWER_CHARS} символов\n"
        f"Текст:\n{text.strip()}"
    )


def _limits_block() -> str:
    return (
        f"- summary не длиннее {MAX_SUMMARY_CHARS} символов\n"
        f"- key_points содержит ровно {KEY_POINT_COUNT} непустые строки, "
        f"каждая не длиннее {MAX_KEY_POINT_CHARS} символов\n"
        f"- final_answer не длиннее {MAX_FINAL_ANSWER_CHARS} символов\n"
        "- category: question, information, advice, bug или other\n"
        "- sentiment: positive, neutral или negative"
    )


VARIANTS: dict[str, PromptVariant] = {
    "plain": PromptVariant(
        name="plain",
        system=f"""\
Ты обрабатываешь пользовательский текст и возвращаешь только JSON-объект.
Не добавляй пояснения и не оборачивай ответ в markdown.

Схема:
{_JSON_SHAPE}

Правила:
- key_points содержит ровно {KEY_POINT_COUNT} непустые строки
- summary — краткое содержание в 1-2 предложениях
- final_answer — короткий полезный ответ
- category: question (вопрос), information (факт или новость), advice (просьба о совете), bug (сбой или жалоба), other (заметка и всё остальное)
- sentiment: positive, neutral или negative
- пиши на том же языке, что и входной текст
- не добавляй другие поля
""",
        build_user=_plain_user,
    ),
    "limits": PromptVariant(
        name="limits",
        system=f"""\
Ты обрабатываешь пользовательский текст и возвращаешь только JSON-объект.
Не добавляй пояснения и не оборачивай ответ в markdown.

Схема:
{_JSON_SHAPE}

Правила:
{_limits_block()}
- пиши на том же языке, что и входной текст
- не добавляй другие поля
""",
        build_user=_plain_user,
    ),
    "checklist": PromptVariant(
        name="checklist",
        system=f"""\
Ты редактор структурированных ответов. Верни только JSON-объект без markdown и пояснений.

Перед ответом проверь чеклист:
{_limits_block()}
- язык ответа совпадает с языком входного текста
- в JSON только поля summary, category, sentiment, key_points и final_answer

Схема:
{_JSON_SHAPE}
""",
        build_user=_checklist_user,
    ),
}


def get_variant(name: str | None = None) -> PromptVariant:
    """Return a prompt variant. The default is the day-2 working prompt, not a comparison winner."""
    key = name or DEFAULT_VARIANT_NAME
    try:
        return VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(VARIANTS)
        raise ValueError(f"Unknown prompt {key!r}. Known prompts: {known}") from exc
