"""CLI for the text pipeline: meaning, category, routed answer, and a self-check."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from json import JSONDecodeError
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llm_client import EmptyModelResponseError, LLMClient, LLMClientError
from prompts import (
    ANSWER_SCHEMA,
    CLASSIFY_SCHEMA,
    DEFAULT_VARIANT_NAME,
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM,
    FALLBACK_SYSTEM,
    MAX_CHECK_NOTES_CHARS,
    MAX_INTENT_CHARS,
    MAX_KEY_POINT_CHARS,
    MAX_SUMMARY_CHARS,
    SELF_CHECK_SCHEMA,
    SELF_CHECK_SYSTEM,
    VARIANTS,
    PromptVariant,
    build_answer_user,
    build_classify_user,
    build_extract_user,
    build_repair_user,
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

M = TypeVar("M", bound=BaseModel)

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
    if isinstance(exc, EmptyModelResponseError):
        return "Модель вернула пустой ответ."
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"]) or "ответ"
            err_type = err["type"]
            if err_type == "missing":
                parts.append(f"нет поля {loc}")
            elif err_type == "string_too_long":
                limit = err.get("ctx", {}).get("max_length")
                if isinstance(limit, int):
                    parts.append(f"поле {loc} длиннее {limit} символов")
                else:
                    parts.append(f"поле {loc} слишком длинное")
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


def _length_only(exc: Exception) -> bool:
    """True when every schema problem is a string that is too long."""
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        return bool(errors) and all(err["type"] == "string_too_long" for err in errors)
    return isinstance(exc, ValueError) and "длиннее лимита маршрута" in str(exc)


def _clip_text(value: object, limit: int) -> object:
    if not isinstance(value, str):
        return value
    return value.strip()[:limit].rstrip()


def clip_meaning(data: dict) -> dict:
    """Shorten meaning fields down to the existing limits."""
    clipped = dict(data)
    clipped["meaning"] = _clip_text(clipped.get("meaning"), MAX_SUMMARY_CHARS)
    points = clipped.get("key_points")
    if isinstance(points, list):
        clipped["key_points"] = [
            _clip_text(item, MAX_KEY_POINT_CHARS) for item in points
        ]
    return clipped


def clip_classification(data: dict) -> dict:
    """Shorten intent down to the existing limit."""
    clipped = dict(data)
    clipped["intent"] = _clip_text(clipped.get("intent"), MAX_INTENT_CHARS)
    return clipped


def clip_final_answer(data: dict, limit: int) -> dict:
    """Shorten the routed answer down to that route's limit."""
    clipped = dict(data)
    clipped["final_answer"] = _clip_text(clipped.get("final_answer"), limit)
    return clipped


def clip_self_check(data: dict) -> dict:
    """Shorten check notes down to the existing limit."""
    clipped = dict(data)
    clipped["notes"] = _clip_text(clipped.get("notes"), MAX_CHECK_NOTES_CHARS)
    return clipped


def _validate_final_answer(data: object, limit: int) -> FinalAnswer:
    answer = FinalAnswer.model_validate(data)
    if len(answer.final_answer) > limit:
        raise ValueError(f"поле final_answer: длиннее лимита маршрута {limit}")
    return answer


def _clipped_summary(summary: str, limit: int) -> str:
    text = summary.strip()[:limit].rstrip()
    if text:
        return text
    fallback = "Не удалось сформировать ответ."
    return fallback[:limit]


def _degraded_check(exc: Exception) -> SelfCheck:
    notes = f"проверка не выполнена: {explain_failure(exc)}"
    if len(notes) > MAX_CHECK_NOTES_CHARS:
        notes = notes[:MAX_CHECK_NOTES_CHARS].rstrip()
    return SelfCheck(consistent=False, details_kept=False, notes=notes)


def request_model(
    client: LLMClient,
    *,
    system: str,
    user: str,
    schema: str,
    step: str,
    warnings: list[str],
    validate: Callable[[object], M],
    clip: Callable[[dict], dict],
) -> M:
    """Call the model once, then one repair prompt. Clip only a still-too-long repair."""
    raw = client.complete(system, user)
    try:
        return validate(parse_json_object(raw))
    except (JSONDecodeError, ValidationError, ValueError) as exc:
        reason = explain_failure(exc)
        logger.warning("step=%s fallback reason=%s", step, reason)
        warnings.append(f"fallback: {step}: {reason}")
        repaired = client.complete(
            FALLBACK_SYSTEM,
            build_repair_user(schema, reason, raw),
        )
        try:
            return validate(parse_json_object(repaired))
        except (JSONDecodeError, ValidationError, ValueError) as repair_exc:
            if not _length_only(repair_exc):
                raise
            try:
                parsed = parse_json_object(repaired)
            except JSONDecodeError:
                raise repair_exc from None
            if not isinstance(parsed, dict):
                raise repair_exc
            try:
                value = validate(clip(parsed))
            except (ValidationError, ValueError) as clip_exc:
                raise repair_exc from clip_exc
            logger.warning("step=%s truncated", step)
            warnings.append(f"truncated: {step}")
            return value


def extract_meaning(
    client: LLMClient,
    text: str,
    warnings: list[str],
) -> MeaningExtract:
    """Step 1. Sees only the raw text."""
    logger.info("step=extract_meaning start chars=%s", len(text))
    extracted = request_model(
        client,
        system=EXTRACT_SYSTEM,
        user=build_extract_user(text),
        schema=EXTRACT_SCHEMA,
        step="extract_meaning",
        warnings=warnings,
        validate=MeaningExtract.model_validate,
        clip=clip_meaning,
    )
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
    warnings: list[str],
) -> RequestClassification:
    """Step 2. Sees the extracted meaning and key points."""
    logger.info("step=classify start prompt=%s", variant.name)
    classification = request_model(
        client,
        system=variant.system,
        user=build_classify_user(
            text,
            extracted.meaning,
            extracted.key_points,
            variant,
        ),
        schema=CLASSIFY_SCHEMA,
        step="classify",
        warnings=warnings,
        validate=RequestClassification.model_validate,
        clip=clip_classification,
    )
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
    warnings: list[str],
) -> str:
    """Step 4. The route and the answer both come from the structured fields."""
    route = get_route(fields.category.value)
    logger.info("step=final_answer start route=%s", route.name)
    try:
        answer = request_model(
            client,
            system=route.system,
            user=build_answer_user(
                text,
                fields.summary,
                fields.intent,
                fields.key_points,
                fields.category.value,
            ),
            schema=ANSWER_SCHEMA,
            step="final_answer",
            warnings=warnings,
            validate=lambda data: _validate_final_answer(data, route.max_chars),
            clip=lambda data: clip_final_answer(data, route.max_chars),
        )
    except (LLMClientError, ValidationError, ValueError) as exc:
        message = explain_failure(exc)
        logger.warning(
            "step=final_answer degraded error=%s: %s",
            type(exc).__name__,
            message,
        )
        warnings.append(f"degraded: final_answer: {message}")
        final_answer = _clipped_summary(fields.summary, route.max_chars)
        logger.info(
            "step=final_answer done route=%s chars=%s degraded=yes",
            route.name,
            len(final_answer),
        )
        return final_answer
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
    warnings: list[str],
) -> SelfCheck:
    """Step 5. Checks the answer against the original text and earlier fields."""
    logger.info("step=self_check start")
    try:
        check = request_model(
            client,
            system=SELF_CHECK_SYSTEM,
            user=build_self_check_user(
                text,
                fields.summary,
                fields.key_points,
                fields.category.value,
                fields.intent,
                final_answer,
            ),
            schema=SELF_CHECK_SCHEMA,
            step="self_check",
            warnings=warnings,
            validate=SelfCheck.model_validate,
            clip=clip_self_check,
        )
    except (LLMClientError, ValidationError, ValueError) as exc:
        message = explain_failure(exc)
        logger.warning(
            "step=self_check degraded error=%s: %s",
            type(exc).__name__,
            message,
        )
        warnings.append(f"degraded: self_check: {message}")
        check = _degraded_check(exc)
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
    warnings: list[str] = []
    logger.info("chain start length=%s prompt=%s", len(text), selected.name)
    extracted = extract_meaning(client, text, warnings)
    classification = classify_request(client, text, extracted, selected, warnings)
    fields = build_structured_fields(extracted, classification)
    final_answer = generate_final_answer(client, text, fields, warnings)
    check = self_check_result(client, text, fields, final_answer, warnings)
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
        warnings=warnings,
    )
    logger.info(
        "chain done category=%s route=%s consistent=%s details_kept=%s warnings=%s",
        result.category.value,
        result.category.value,
        result.consistent,
        result.details_kept,
        len(result.warnings),
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
        scenario = str(item.get("scenario") or "").strip()
        if scenario:
            example["scenario"] = scenario
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
        if item.get("scenario"):
            record["scenario"] = item["scenario"]
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
        scenario = record.get("scenario")
        if scenario:
            print(f"=== {record['id']}: {scenario} ({prompt_name}) ===")
        else:
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
            if record.get("warnings"):
                print("Предупреждения: " + "; ".join(record["warnings"]))
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
    """Print category accuracy. The 10-example set fails the run below 8 matches."""
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
    if len(expected) < 10:
        return 0
    return 1 if len(matched) < 8 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разобрать текст: смысл, категория, ответ по маршруту и проверка.",
    )
    parser.add_argument("--text", help="Обработать один произвольный текст.")
    parser.add_argument(
        "--input",
        type=Path,
        default=EXAMPLES_PATH,
        help=f"JSON со списком примеров (по умолчанию {EXAMPLES_PATH}).",
    )
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
            items = load_examples(args.input)
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
