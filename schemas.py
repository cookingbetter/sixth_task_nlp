"""Structured result of the text processing step."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from prompts import (
    KEY_POINT_COUNT,
    MAX_FINAL_ANSWER_CHARS,
    MAX_KEY_POINT_CHARS,
    MAX_SUMMARY_CHARS,
)


class Category(str, Enum):
    question = "question"
    information = "information"
    advice = "advice"
    bug = "bug"
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


KeyPoint = Annotated[str, Field(min_length=1, max_length=MAX_KEY_POINT_CHARS)]


class TextProcessingResult(BaseModel):
    """Structured reading of one user text."""

    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    category: Category
    sentiment: Sentiment
    key_points: list[KeyPoint] = Field(
        min_length=KEY_POINT_COUNT,
        max_length=KEY_POINT_COUNT,
    )
    final_answer: str = Field(min_length=1, max_length=MAX_FINAL_ANSWER_CHARS)

    @field_validator("summary", "final_answer", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def strip_label(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("key_points", mode="before")
    @classmethod
    def strip_key_points(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        cleaned: list[object] = []
        for item in value:
            if not isinstance(item, str):
                cleaned.append(item)
                continue
            text = item.strip()
            if not text:
                raise ValueError("key points must not be blank")
            cleaned.append(text)
        return cleaned
