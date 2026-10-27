"""Prompt templates for the day-1 text processing step."""

SYSTEM_PROMPT = """\
Ты обрабатываешь пользовательский текст и возвращаешь только JSON-объект.
Не добавляй пояснения и не оборачивай ответ в markdown.

Схема:
{
  "summary": "краткое содержание в 1-2 предложениях",
  "key_points": ["мысль 1", "мысль 2", "мысль 3"],
  "response": "короткий полезный ответ или следующий шаг"
}

Правила:
- key_points содержит ровно 3 непустые строки
- summary и response не пустые
- пиши на том же языке, что и входной текст
- не добавляй другие поля
"""


def build_user_prompt(text: str) -> str:
    """Build the user message for one input text."""
    return f"Текст:\n{text.strip()}"
