"""Prompt variants and category routes for the text processing step."""

from collections.abc import Callable
from dataclasses import dataclass

MAX_SUMMARY_CHARS = 280
MAX_KEY_POINT_CHARS = 120
MAX_FINAL_ANSWER_CHARS = 200
MAX_SALES_ANSWER_CHARS = 160
MAX_INTENT_CHARS = 120
MAX_CHECK_NOTES_CHARS = 200
KEY_POINT_COUNT = 3

DEFAULT_VARIANT_NAME = "plain"

ROUTE_ANSWER_LIMITS = {
    "support": MAX_FINAL_ANSWER_CHARS,
    "feedback": MAX_FINAL_ANSWER_CHARS,
    "complaint": MAX_FINAL_ANSWER_CHARS,
    "sales": MAX_SALES_ANSWER_CHARS,
    "general_question": MAX_FINAL_ANSWER_CHARS,
}

_CLASSIFY_SHAPE = """\
{
  "category": "support | feedback | complaint | sales | general_question",
  "sentiment": "positive | neutral | negative",
  "intent": "чего хочет автор текста"
}"""

CLASSIFY_SCHEMA = _CLASSIFY_SHAPE

EXTRACT_SCHEMA = """\
{
  "meaning": "что сказано в тексте, 1-2 предложения",
  "key_points": ["мысль 1", "мысль 2", "мысль 3"]
}"""

ANSWER_SCHEMA = """\
{"final_answer": "ответ пользователю"}"""

SELF_CHECK_SCHEMA = """\
{
  "consistent": true,
  "details_kept": true,
  "notes": "одно короткое предложение"
}"""

MAX_REPAIR_CHARS = 500

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
        "Ограничения классификации:\n"
        f"- intent не длиннее {MAX_INTENT_CHARS} символов\n"
        "- category и sentiment только из списка в system prompt\n"
        f"Текст:\n{text.strip()}"
    )


def _limits_block() -> str:
    return (
        f"- intent не длиннее {MAX_INTENT_CHARS} символов\n"
        f"{_CATEGORY_RULES}"
    )


def _answer_system(instruction: str, max_chars: int) -> str:
    return f"""\
Ты пишешь только итоговый ответ и возвращаешь JSON-объект.
Не добавляй пояснения и не оборачивай ответ в markdown.

Схема:
{ANSWER_SCHEMA}

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
Ты классифицируешь уже извлечённый смысл и возвращаешь только JSON-объект.
Не пиши итоговый ответ и не оборачивай JSON в markdown.
Опирайся на извлечённый смысл и ключевые мысли, а не пересказывай текст заново.

Схема:
{_CLASSIFY_SHAPE}

Правила:
{_CATEGORY_RULES}
- пиши на том же языке, что и входной текст
- не добавляй другие поля
""",
        build_user=_plain_user,
    ),
    "limits": PromptVariant(
        name="limits",
        system=f"""\
Ты классифицируешь уже извлечённый смысл и возвращаешь только JSON-объект.
Не пиши итоговый ответ и не оборачивай JSON в markdown.
Опирайся на извлечённый смысл и ключевые мысли, а не пересказывай текст заново.

Схема:
{_CLASSIFY_SHAPE}

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
- в JSON только поля category, sentiment и intent

Схема:
{_CLASSIFY_SHAPE}
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


EXTRACT_SYSTEM = f"""\
Ты извлекаешь смысл текста и возвращаешь только JSON-объект.
Не классифицируй запрос и не пиши ответ пользователю.

Схема:
{EXTRACT_SCHEMA}

Правила:
- meaning не длиннее {MAX_SUMMARY_CHARS} символов
- key_points содержит ровно {KEY_POINT_COUNT} непустые строки, каждая не длиннее {MAX_KEY_POINT_CHARS} символов
- сохрани имена, числа и конкретную просьбу, если они есть
- пиши на том же языке, что и входной текст
- не добавляй другие поля
"""

SELF_CHECK_SYSTEM = f"""\
Ты проверяешь готовый ответ и возвращаешь только JSON-объект.
Не переписывай ответ.

Схема:
{SELF_CHECK_SCHEMA}

Правила:
- consistent равен false, если ответ противоречит исходному тексту или смыслу
- details_kept равен false, если из ответа пропала важная деталь: число, имя, конкретная просьба или факт
- notes не длиннее {MAX_CHECK_NOTES_CHARS} символов
- consistent и details_kept только булевы true или false
- не добавляй другие поля
"""

FALLBACK_SYSTEM = """\
Предыдущий ответ не прошёл проверку. Верни только один JSON-объект по схеме из сообщения пользователя.
Без markdown, без пояснений и без других полей.
Если строка длиннее лимита, сократи её.
"""


def build_repair_user(schema: str, reason: str, raw: str) -> str:
    """Ask for one schema-valid JSON object instead of the rejected answer."""
    bad = raw.strip()
    if len(bad) > MAX_REPAIR_CHARS:
        bad = bad[:MAX_REPAIR_CHARS].rstrip()
    return (
        f"Схема:\n{schema.strip()}\n"
        f"Почему ответ отклонён: {reason.strip()}\n"
        f"Плохой ответ:\n{bad}"
    )


def build_extract_user(text: str) -> str:
    """User message for the meaning step. It sees only the raw text."""
    return f"Текст:\n{text.strip()}"


def build_classify_user(
    text: str,
    meaning: str,
    key_points: list[str],
    variant: PromptVariant,
) -> str:
    """User message for classification. It sees the extracted meaning first."""
    points = "\n".join(f"- {point}" for point in key_points)
    return (
        f"Извлечённый смысл:\n{meaning}\n"
        f"Ключевые мысли:\n{points}\n"
        f"{variant.build_user(text)}"
    )


def build_self_check_user(
    text: str,
    summary: str,
    key_points: list[str],
    category: str,
    intent: str,
    final_answer: str,
) -> str:
    """User message for the check step. It sees the text, fields, and answer."""
    points = "\n".join(f"- {point}" for point in key_points)
    return (
        f"Исходный текст:\n{text.strip()}\n"
        f"Смысл:\n{summary}\n"
        f"Категория: {category}\n"
        f"Intent: {intent}\n"
        f"Ключевые мысли:\n{points}\n"
        f"Ответ:\n{final_answer}"
    )


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
