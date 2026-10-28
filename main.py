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
    EXTRACT_SYSTEM,
    SELF_CHECK_SYSTEM,
    VARIANTS,
    PromptVariant,
    build_answer_user,
    build_classify_user,
    build_extract_user,
    build_self_check_user,
    get_route,
    get_variant,
)
from schemas import (
    Category,
    FinalAnswer,
    MeaningExtract,
    RequestClassification,
    SelfCheck,
    Sentiment,
    StructuredFields,
    TextProcessingResult,
)

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


def extract_meaning(client: LLMClient, text: str) -> MeaningExtract:
    """Step 1. Sees only the raw text."""
    logger.info("step=extract_meaning start chars=%s", len(text))
    raw = client.complete(EXTRACT_SYSTEM, build_extract_user(text))
    extracted = MeaningExtract.model_validate(parse_json_object(raw))
    logger.info(
        "step=extract_meaning done meaning_chars=%s points=%s",
        len(extracted.meaning),
        len(extracted.key_points),
    )
    return extracted


def classify_request(
    client: LLMClient,
    text: str,
    extracted: MeaningExtract,
    variant: PromptVariant,
) -> RequestClassification:
    """Step 2. Sees the extracted meaning and key points."""
    logger.info("step=classify start prompt=%s", variant.name)
    raw = client.complete(
        variant.system,
        build_classify_user(text, extracted.meaning, extracted.key_points, variant),
    )
    classification = RequestClassification.model_validate(parse_json_object(raw))
    logger.info(
        "step=classify done category=%s sentiment=%s intent=%s",
        classification.category.value,
        classification.sentiment.value,
        classification.intent,
    )
    return classification


def build_structured_fields(
    extracted: MeaningExtract,
    classification: RequestClassification,
) -> StructuredFields:
    """Step 3. Assembles one object from the two previous steps. No model call."""
    fields = StructuredFields(
        summary=extracted.meaning,
        key_points=extracted.key_points,
        category=classification.category,
        sentiment=classification.sentiment,
        intent=classification.intent,
    )
    logger.info(
        "step=structured_fields category=%s sentiment=%s points=%s",
        fields.category.value,
        fields.sentiment.value,
        len(fields.key_points),
    )
    return fields


def generate_final_answer(
    client: LLMClient,
    text: str,
    fields: StructuredFields,
) -> str:
    """Step 4. The route and the answer both come from the structured fields."""
    route = get_route(fields.category.value)
    logger.info("step=final_answer start route=%s", route.name)
    raw = client.complete(
        route.system,
        build_answer_user(
            text,
            fields.summary,
            fields.intent,
            fields.key_points,
            fields.category.value,
        ),
    )
    answer = FinalAnswer.model_validate(parse_json_object(raw))
    logger.info(
        "step=final_answer done route=%s chars=%s",
        route.name,
        len(answer.final_answer),
    )
    return answer.final_answer


def self_check_result(
    client: LLMClient,
    text: str,
    fields: StructuredFields,
    final_answer: str,
) -> SelfCheck:
    """Step 5. Checks the answer against the original text and earlier fields."""
    logger.info("step=self_check start")
    raw = client.complete(
        SELF_CHECK_SYSTEM,
        build_self_check_user(
            text,
            fields.summary,
            fields.key_points,
            fields.category.value,
            fields.intent,
            final_answer,
        ),
    )
    check = SelfCheck.model_validate(parse_json_object(raw))
    logger.info(
        "step=self_check done consistent=%s details_kept=%s",
        check.consistent,
        check.details_kept,
    )
    return check


def process_text(
    client: LLMClient,
    text: str,
    variant: PromptVariant | None = None,
) -> TextProcessingResult:
    """Run extract, classify, structured fields, routed answer, and self-check."""
    selected = variant or get_variant()
    logger.info("chain start length=%s prompt=%s", len(text), selected.name)
    extracted = extract_meaning(client, text)
    classification = classify_request(client, text, extracted, selected)
    fields = build_structured_fields(extracted, classification)
    final_answer = generate_final_answer(client, text, fields)
    check = self_check_result(client, text, fields, final_answer)
    result = TextProcessingResult(
        summary=fields.summary,
        key_points=fields.key_points,
        category=fields.category,
        sentiment=fields.sentiment,
        intent=fields.intent,
        final_answer=final_answer,
        consistent=check.consistent,
        details_kept=check.details_kept,
        check_notes=check.notes,
    )
    logger.info(
        "chain done category=%s route=%s consistent=%s details_kept=%s",
        result.category.value,
        result.category.value,
        result.consistent,
        result.details_kept,
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
            print("Шаги:")
            print(f"1. extract_meaning: {record['summary']}")
            print(
                "2. classify: "
                f"category={record['category']} "
                f"sentiment={record['sentiment']} "
                f"intent={record['intent']}"
            )
            points = "; ".join(record["key_points"])
            print(f"3. structured_fields: {points}")
            print(f"4. final_answer route={record['route']}: {record['final_answer']}")
            print(
                "5. self_check: "
                f"consistent={record['consistent']} "
                f"details_kept={record['details_kept']} "
                f"notes={record['check_notes']}"
            )
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
