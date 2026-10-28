"""Pipeline: one text in, summary, key points, and a short response out."""

from __future__ import annotations

import argparse
import json
import logging
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from llm_client import LLMClient, LLMClientError
from prompts import DEFAULT_VARIANT_NAME, VARIANTS, PromptVariant, get_variant
from schemas import TextProcessingResult

EXAMPLES_PATH = Path("examples/sample_inputs.json")
DEFAULT_OUTPUT = Path("output/results.json")

logger = logging.getLogger("pipeline")


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


def process_text(
    client: LLMClient,
    text: str,
    variant: PromptVariant | None = None,
) -> TextProcessingResult:
    """Call the model and return a validated processing result."""
    selected = variant or get_variant()
    logger.info("processing text length=%s prompt=%s", len(text), selected.name)
    raw = client.complete(selected.system, selected.build_user(text))
    logger.info("model response received chars=%s", len(raw))

    try:
        payload = json.loads(strip_json_fences(raw))
    except JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc.msg}") from exc

    result = TextProcessingResult.model_validate(payload)
    logger.info("validation ok")
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
        examples.append({"id": item_id, "text": text})
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
        try:
            result = process_text(client, item["text"], selected)
        except (LLMClientError, ValidationError, ValueError) as exc:
            logger.error(
                "text failed id=%s prompt=%s error=%s: %s",
                item["id"],
                selected.name,
                type(exc).__name__,
                exc,
            )
            record["error"] = str(exc)
        else:
            record.update(result.model_dump())
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
            print("Ключевые мысли:")
            for index, point in enumerate(record["key_points"], start=1):
                print(f"{index}. {point}")
            print(f"Ответ: {record['response']}")
        print()


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("results saved path=%s", path)


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
    save_results(args.output, results)
    return 1 if any("error" in record for record in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
