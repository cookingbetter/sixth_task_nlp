# sixth_task_nlp

Учебный LLM-пайплайн для обработки текста. День 1 принимает сырой текст и возвращает краткое содержание, три ключевые мысли и короткий полезный ответ. День 2 сравнивает три формулировки промпта с явными пределами длины.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите `OPENAI_API_KEY`. Для совместимого API можно задать `OPENAI_BASE_URL` и `OPENAI_MODEL`.

## Запуск

Все примеры из `examples/sample_inputs.json`:

```bash
python main.py
```

Один произвольный текст:

```bash
python main.py --text "Как подготовиться к собеседованию на junior Python?"
```

Свой файл результата (по умолчанию `output/results.json`):

```bash
python main.py --output output/results.json
```

В консоли каждый текст печатается отдельным блоком. В JSON сохраняется список объектов с исходным текстом и полями `summary`, `key_points`, `response`. Если модель вернула пустой ответ, невалидный JSON или API ответило ошибкой, для этого текста записывается поле `error`, а остальные примеры продолжают обрабатываться.

По умолчанию используется промпт `plain`. Другой вариант:

```bash
python main.py --prompt plain
python main.py --prompt checklist
```

Сравнение трёх формулировок на одних и тех же примерах:

```bash
python compare_prompts.py
```

Результат пишется в `output/prompt_comparison.json`. Какой вариант лучше, фиксируется в `prompt_comparison.md` после прогона с ключом.
