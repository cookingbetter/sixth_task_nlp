"""Prompt variants and category routes for the text processing step."""

from collections.abc import Callable
from dataclasses import dataclass

MAX_SUMMARY_CHARS = 280
MAX_KEY_POINT_CHARS = 120
MAX_FINAL_ANSWER_CHARS = 200
MAX_SALES_ANSWER_CHARS = 160
MAX_INTENT_CHARS = 120
KEY_POINT_COUNT = 3

DEFAULT_VARIANT_NAME = "plain"

ROUTE_ANSWER_LIMITS = {
    "support": MAX_FINAL_ANSWER_CHARS,
    "feedback": MAX_FINAL_ANSWER_CHARS,
    "complaint": MAX_FINAL_ANSWER_CHARS,
    "sales": MAX_SALES_ANSWER_CHARS,
    "general_question": MAX_FINAL_ANSWER_CHARS,
}

_JSON_SHAPE = """\
{
  "summary": "краткое содержание",
  "category": "support | feedback | complaint | sales | general_question",
  "sentiment": "positive | neutral | negative",
  "intent": "чего хочет автор текста",
  "key_points": ["мысль 1", "мысль 2", "мысль 3"]
}"""

_CATEGORY_RULES = """\
- category только одно из значений:
  support — сбой, ошибка, как починить или где нажать
  feedback — отзыв, похвала или пожелание без требования вернуть деньги
  complaint — претензия, возврат, повторный сбой, недовольство сервисом
  sales — цена, тариф, сравнение, почему купить
  general_question — нейтральный вопрос без поломки и без покупки
- sentiment: positive, neutral или negative
- intent — короткая цель автора, не длиннее 120 символов"""


@dataclass(frozen=True)
class PromptVariant:
    """System prompt and user-message template for classification."""

    name: str
    system: str
    build_user: Callable[[str], str]


@dataclass(frozen=True)
class AnswerRoute:
    """One category's answer instruction. Selection happens outside the prompt."""

    name: str
    system: str
    max_chars: int


def _plain_user(text: str) -> str:
    return f"Текст:\n{text.strip()}"


def _checklist_user(text: str) -> str:
    return (
        "Ограничения:\n"
        f"- summary не длиннее {MAX_SUMMARY_CHARS} символов\n"
        f"- key_points: ровно {KEY_POINT_COUNT} пункта, каждый не длиннее {MAX_KEY_POINT_CHARS} символов\n"
        f"- intent не длиннее {MAX_INTENT_CHARS} символов\n"
        f"Текст:\n{text.strip()}"
    )


def _limits_block() -> str:
    return (
        f"- summary не длиннее {MAX_SUMMARY_CHARS} символов\n"
        f"- key_points содержит ровно {KEY_POINT_COUNT} непустые строки, "
        f"каждая не длиннее {MAX_KEY_POINT_CHARS} символов\n"
        f"- intent не длиннее {MAX_INTENT_CHARS} символов\n"
        f"{_CATEGORY_RULES}"
    )


def _answer_system(instruction: str, max_chars: int) -> str:
    return f"""\
Ты пишешь только итоговый ответ и возвращаешь JSON-объект.
Не добавляй пояснения и не оборачивай ответ в markdown.

Схема:
{{"final_answer": "ответ пользователю"}}

Инструкция:
{instruction}
- final_answer не длиннее {max_chars} символов
- пиши на том же языке, что и входной текст
- не добавляй другие поля
"""


VARIANTS: dict[str, PromptVariant] = {
    "plain": PromptVariant(
        name="plain",
        system=f"""\
Ты классифицируешь пользовательский текст и возвращаешь только JSON-объект.
Не пиши итоговый ответ и не оборачивай JSON в markdown.

Схема:
{_JSON_SHAPE}

Правила:
- key_points содержит ровно {KEY_POINT_COUNT} непустые строки
- summary — краткое содержание в 1-2 предложениях
{_CATEGORY_RULES}
- пиши на том же языке, что и входной текст
- не добавляй другие поля
""",
        build_user=_plain_user,
    ),
    "limits": PromptVariant(
        name="limits",
        system=f"""\
Ты классифицируешь пользовательский текст и возвращаешь только JSON-объект.
Не пиши итоговый ответ и не оборачивай JSON в markdown.

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
Ты редактор классификации. Верни только JSON-объект без markdown и пояснений.
Не пиши итоговый ответ.

Перед ответом проверь чеклист:
{_limits_block()}
- язык ответа совпадает с языком входного текста
- в JSON только поля summary, category, sentiment, intent и key_points

Схема:
{_JSON_SHAPE}
""",
        build_user=_checklist_user,
    ),
}

ROUTES: dict[str, AnswerRoute] = {
    "complaint": AnswerRoute(
        name="complaint",
        max_chars=ROUTE_ANSWER_LIMITS["complaint"],
        system=_answer_system(
            "Ответь спокойно и эмпатично. Признай проблему своими словами и предложи один конкретный следующий шаг. Не продавай и не читай лекцию.",
            ROUTE_ANSWER_LIMITS["complaint"],
        ),
    ),
    "sales": AnswerRoute(
        name="sales",
        max_chars=ROUTE_ANSWER_LIMITS["sales"],
        system=_answer_system(
            "Дай короткий продающий ответ: одна выгода и один призыв к действию. Без технических шагов и без длинных извинений.",
            ROUTE_ANSWER_LIMITS["sales"],
        ),
    ),
    "support": AnswerRoute(
        name="support",
        max_chars=ROUTE_ANSWER_LIMITS["support"],
        system=_answer_system(
            "Дай структурированный технический ответ из 2-3 коротких шагов проверки. Без продажи и без эмоциональных извинений.",
            ROUTE_ANSWER_LIMITS["support"],
        ),
    ),
    "feedback": AnswerRoute(
        name="feedback",
        max_chars=ROUTE_ANSWER_LIMITS["feedback"],
        system=_answer_system(
            "Поблагодари за отзыв и напиши, что именно возьмём в работу. Без продажи.",
            ROUTE_ANSWER_LIMITS["feedback"],
        ),
    ),
    "general_question": AnswerRoute(
        name="general_question",
        max_chars=ROUTE_ANSWER_LIMITS["general_question"],
        system=_answer_system(
            "Дай прямой ответ на вопрос. Без продажи и без лишней эмпатии.",
            ROUTE_ANSWER_LIMITS["general_question"],
        ),
    ),
}


def get_variant(name: str | None = None) -> PromptVariant:
    """Return a classification prompt variant."""
    key = name or DEFAULT_VARIANT_NAME
    try:
        return VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(VARIANTS)
        raise ValueError(f"Unknown prompt {key!r}. Known prompts: {known}") from exc


def get_route(category: str) -> AnswerRoute:
    """Return the answer route selected by category. Unknown categories have no route."""
    try:
        return ROUTES[category]
    except KeyError as exc:
        raise ValueError(f"Нет маршрута для категории {category}") from exc


def build_answer_user(
    text: str,
    summary: str,
    intent: str,
    key_points: list[str],
    category: str,
) -> str:
    """Build the user message for the routed answer call."""
    points = "\n".join(f"- {point}" for point in key_points)
    return (
        f"Категория: {category}\n"
        f"Intent: {intent}\n"
        f"Summary: {summary}\n"
        f"Ключевые мысли:\n{points}\n"
        f"Текст:\n{text.strip()}"
    )
