"""Run every prompt variant on the same demo texts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from llm_client import LLMClient, LLMClientError
from main import EXAMPLES_PATH, explain_failure, load_examples, process_text, save_results
from prompts import VARIANTS, get_variant

DEFAULT_OUTPUT = Path("output/prompt_comparison.json")

logger = logging.getLogger("pipeline")


def compare_prompts(client: LLMClient, items: list[dict[str, str]]) -> list[dict]:
    """Call each variant on each text. One failure does not stop the rest."""
    records: list[dict] = []
    for name in VARIANTS:
        variant = get_variant(name)
        for item in items:
            record: dict = {"prompt": name, "id": item["id"]}
            try:
                result = process_text(client, item["text"], variant)
            except (LLMClientError, ValidationError, ValueError) as exc:
                message = explain_failure(exc)
                logger.error(
                    "text failed id=%s prompt=%s error=%s: %s",
                    item["id"],
                    name,
                    type(exc).__name__,
                    message,
                )
                record["ok"] = False
                record["error"] = message
            else:
                dumped = result.model_dump(mode="json")
                record["ok"] = True
                record["summary_chars"] = len(dumped["summary"])
                record["key_point_chars"] = [len(point) for point in dumped["key_points"]]
                record["final_answer_chars"] = len(dumped["final_answer"])
                record.update(dumped)
            records.append(record)
    return records


def print_comparison(records: list[dict]) -> None:
    """Print one line per prompt and text."""
    for record in records:
        label = f"{record['prompt']:10} {record['id']:10}"
        if record["ok"]:
            points = ",".join(str(length) for length in record["key_point_chars"])
            print(
                f"{label} ok "
                f"summary={record['summary_chars']} "
                f"points={points} "
                f"final_answer={record['final_answer_chars']}"
            )
        else:
            print(f"{label} error {record['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить формулировки промпта на одних и тех же текстах.",
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
        items = load_examples(EXAMPLES_PATH)
    except (LLMClientError, OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("compare setup failed error=%s: %s", type(exc).__name__, exc)
        return 1

    records = compare_prompts(client, items)
    print_comparison(records)
    save_results(args.output, records)
    failed = sum(1 for record in records if not record["ok"])
    logger.info("compare finished ok=%s failed=%s", len(records) - failed, failed)
    return 1 if failed == len(records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
