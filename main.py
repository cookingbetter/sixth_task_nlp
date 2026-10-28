"""Pipeline: one text in, summary, key points, and a short response out."""

from __future__ import annotations

import argparse
import json
import logging
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from llm_client import LLMClient, LLMClientError
from prompts import (
    DEFAULT_VARIANT_NAME,
    VARIANTS,
    PromptVariant,
    build_answer_user,
    get_route,
    get_variant,
)
from schemas import Category, Classification, FinalAnswer, Sentiment, TextProcessingResult

EXAMPLES_PATH = Path("examples/sample_inputs.json")
DEFAULT_OUTPUT = Path("output/results.json")

logger = logging.getLogger("pipeline")

CATEGORY_LABELS = {
    Category.support: "Техническая помощь",
    Category.feedback: "Ответ на отзыв",
    Category.complaint: "Ответ на жалобу",
    Category.sales: "Предложение",
    Category.general_question: "Ответ на вопрос",
}
SENTIMENT_PREFIX = {
    Sentiment.negative: "Сначала разберите проблему. ",
    Sentiment.positive: "Можно опереться на удачную часть. ",
    Sentiment.neutral: "",
}
_TYPE_ERRORS = {
    "enum",
    "literal_error",
    "is_instance_of",
    "model_type",
}


def build_delivery(result: TextProcessingResult) -> str:
    """Build the user-facing answer from category, sentiment, and final_answer."""
    label = CATEGORY_LABELS[result.category]
    prefix = SENTIMENT_PREFIX[result.sentiment]
    return f"{label}: {prefix}{result.final_answer}"


def explain_failure(exc: Exception) -> str:
    """Turn a parse or schema error into one short message."""
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"]) or "ответ"
            err_type = err["type"]
            if err_type == "missing":
                parts.append(f"нет поля {loc}")
            elif err_type in _TYPE_ERRORS or err_type.endswith("_type"):
                parts.append(f"поле {loc} имеет неверный тип")
            else:
                parts.append(f"поле {loc}: {err['msg']}")
        return "Ответ не прошёл схему: " + "; ".join(parts)
    if isinstance(exc, JSONDecodeError):
        return (
            f"Невалидный JSON: {exc.msg} "
            f"(строка {exc.lineno}, символ {exc.colno})"
        )
    return str(exc)


def strip_json_fences(raw: str) -> str:
    """Drop a markdown code fence if the model wrapped the JSON."""
    text = raw.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object(raw: str) -> object:
    """Parse model text as JSON, dropping a markdown fence when present."""
    return json.loads(strip_json_fences(raw))


def process_text(
    client: LLMClient,
    text: str,
    variant: PromptVariant | None = None,
) -> TextProcessingResult:
    """Classify the text, then generate an answer with the category route."""
    selected = variant or get_variant()
    logger.info("processing text length=%s prompt=%s", len(text), selected.name)
    raw = client.complete(selected.system, selected.build_user(text))
    logger.info("classification response received chars=%s", len(raw))
    classification = Classification.model_validate(parse_json_object(raw))
    route = get_route(classification.category.value)
    logger.info(
        "route selected route=%s intent=%s",
        route.name,
        classification.intent,
    )

    answer_raw = client.complete(
        route.system,
        build_answer_user(
            text,
            classification.summary,
            classification.intent,
            classification.key_points,
            classification.category.value,
        ),
    )
    logger.info("answer response received chars=%s", len(answer_raw))
    answer = FinalAnswer.model_validate(parse_json_object(answer_raw))
    result = TextProcessingResult(
        summary=classification.summary,
        category=classification.category,
        sentiment=classification.sentiment,
        intent=classification.intent,
        key_points=classification.key_points,
        final_answer=answer.final_answer,
    )
    logger.info(
        "validation ok category=%s sentiment=%s route=%s",
        result.category.value,
        result.sentiment.value,
        route.name,
    )
    return result


def load_examples(path: Path) -> list[dict[str, str]]:
    """Load demo texts. Each item has an id and a text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty list of examples.")

    examples: list[dict[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Example {index} must be an object.")
        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError(f"Example {index} has an empty text.")
        item_id = str(item.get("id") or f"example-{index}")
        example = {"id": item_id, "text": text}
        expected = str(item.get("expected_category") or "").strip()
        if expected:
            example["expected_category"] = expected
        examples.append(example)
    return examples


def run_items(
    client: LLMClient,
    items: list[dict[str, str]],
    variant: PromptVariant | None = None,
) -> list[dict]:
    """Process every item. A failure on one text does not stop the rest."""
    selected = variant or get_variant()
    logger.info(
        "run started count=%s model=%s prompt=%s",
        len(items),
        client.model,
        selected.name,
    )
    results: list[dict] = []

    for item in items:
        record = {"id": item["id"], "text": item["text"], "prompt": selected.name}
        if item.get("expected_category"):
            record["expected_category"] = item["expected_category"]
        try:
            result = process_text(client, item["text"], selected)
        except (LLMClientError, ValidationError, ValueError) as exc:
            message = explain_failure(exc)
            logger.error(
                "text failed id=%s prompt=%s error=%s: %s",
                item["id"],
                selected.name,
                type(exc).__name__,
                message,
            )
            record["error"] = message
        else:
            record.update(result.model_dump(mode="json"))
            record["route"] = result.category.value
            record["delivery"] = build_delivery(result)
        results.append(record)

    failed = sum(1 for record in results if "error" in record)
    logger.info("run finished ok=%s failed=%s", len(results) - failed, failed)
    return results


def print_results(results: list[dict]) -> None:
    """Print a readable block for each processed text."""
    for record in results:
        prompt_name = record.get("prompt", DEFAULT_VARIANT_NAME)
        print(f"=== {record['id']} ({prompt_name}) ===")
        print(record["text"])
        print()
        if "error" in record:
            print(f"Ошибка: {record['error']}")
        else:
            print(f"Summary: {record['summary']}")
            print(f"Категория: {record['category']}")
            print(f"Маршрут: {record['route']}")
            print(f"Intent: {record['intent']}")
            print(f"Тональность: {record['sentiment']}")
            print("Ключевые мысли:")
            for index, point in enumerate(record["key_points"], start=1):
                print(f"{index}. {point}")
            print(record["delivery"])
        print()


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("results saved path=%s", path)


def report_categories(results: list[dict]) -> int:
    """Print how many predicted categories match the examples. Return 1 below 8 matches."""
    expected = [record for record in results if record.get("expected_category")]
    if not expected:
        return 0
    matched = [
        record
        for record in expected
        if record.get("category") == record["expected_category"]
    ]
    print(f"Категории совпали: {len(matched)} из {len(expected)}")
    for record in expected:
        if record.get("category") != record["expected_category"]:
            got = record.get("category", "ошибка")
            print(
                f"- {record['id']}: ожидали {record['expected_category']}, получили {got}"
            )
    return 1 if len(matched) < 8 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сжать текст в summary, три ключевые мысли и короткий ответ.",
    )
    parser.add_argument("--text", help="Обработать один произвольный текст.")
    parser.add_argument(
        "--prompt",
        choices=tuple(VARIANTS),
        default=DEFAULT_VARIANT_NAME,
        help=f"Формулировка промпта (по умолчанию {DEFAULT_VARIANT_NAME}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Куда сохранить JSON (по умолчанию {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    try:
        client = LLMClient()
    except LLMClientError as exc:
        logger.error("client setup failed error=%s: %s", type(exc).__name__, exc)
        return 1

    if args.text:
        items = [{"id": "custom", "text": args.text.strip()}]
        if not items[0]["text"]:
            logger.error("text failed id=custom error=ValueError: empty text")
            return 1
    else:
        try:
            items = load_examples(EXAMPLES_PATH)
        except (OSError, JSONDecodeError, ValueError) as exc:
            logger.error("examples failed error=%s: %s", type(exc).__name__, exc)
            return 1

    results = run_items(client, items, get_variant(args.prompt))
    print_results(results)
    category_code = report_categories(results)
    save_results(args.output, results)
    if any("error" in record for record in results) or category_code:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
