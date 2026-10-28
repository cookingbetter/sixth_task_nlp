"""Structured results of the multi-step text pipeline."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from prompts import (
    KEY_POINT_COUNT,
    MAX_CHECK_NOTES_CHARS,
    MAX_FINAL_ANSWER_CHARS,
    MAX_INTENT_CHARS,
    MAX_KEY_POINT_CHARS,
    MAX_SUMMARY_CHARS,
    ROUTE_ANSWER_LIMITS,
)


class Category(str, Enum):
    support = "support"
    feedback = "feedback"
    complaint = "complaint"
    sales = "sales"
    general_question = "general_question"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


KeyPoint = Annotated[str, Field(min_length=1, max_length=MAX_KEY_POINT_CHARS)]


def _strip_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.strip()


class MeaningExtract(BaseModel):
    """Step 1: what the text says, before any category is chosen."""

    meaning: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    key_points: list[KeyPoint] = Field(
        min_length=KEY_POINT_COUNT,
        max_length=KEY_POINT_COUNT,
    )

    @field_validator("meaning", mode="before")
    @classmethod
    def strip_meaning(cls, value: object) -> object:
        cleaned = _strip_text(value)
        if isinstance(cleaned, str) and not cleaned:
            raise ValueError("must not be blank")
        return cleaned

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


class RequestClassification(BaseModel):
    """Step 2: category and intent, using the extracted meaning."""

    category: Category
    sentiment: Sentiment
    intent: str = Field(min_length=1, max_length=MAX_INTENT_CHARS)

    @field_validator("intent", mode="before")
    @classmethod
    def strip_intent(cls, value: object) -> object:
        cleaned = _strip_text(value)
        if isinstance(cleaned, str) and not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def strip_label(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class StructuredFields(BaseModel):
    """Step 3: one object assembled from meaning and classification."""

    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    key_points: list[KeyPoint] = Field(
        min_length=KEY_POINT_COUNT,
        max_length=KEY_POINT_COUNT,
    )
    category: Category
    sentiment: Sentiment
    intent: str = Field(min_length=1, max_length=MAX_INTENT_CHARS)


class FinalAnswer(BaseModel):
    """Step 4: answer produced by the category route."""

    final_answer: str = Field(min_length=1, max_length=MAX_FINAL_ANSWER_CHARS)

    @field_validator("final_answer", mode="before")
    @classmethod
    def strip_final_answer(cls, value: object) -> object:
        cleaned = _strip_text(value)
        if isinstance(cleaned, str) and not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class SelfCheck(BaseModel):
    """Step 5: whether the answer fits the original text."""

    consistent: bool
    details_kept: bool
    notes: str = Field(default="", max_length=MAX_CHECK_NOTES_CHARS)

    @field_validator("notes", mode="before")
    @classmethod
    def strip_notes(cls, value: object) -> object:
        if value is None:
            return ""
        return _strip_text(value)


class TextProcessingResult(StructuredFields):
    """Full chain result: structured fields, routed answer, and self-check."""

    final_answer: str = Field(min_length=1, max_length=MAX_FINAL_ANSWER_CHARS)
    consistent: bool
    details_kept: bool
    check_notes: str = Field(default="", max_length=MAX_CHECK_NOTES_CHARS)

    @field_validator("final_answer", "check_notes", mode="before")
    @classmethod
    def strip_result_text(cls, value: object) -> object:
        if value is None:
            return ""
        return _strip_text(value)

    @model_validator(mode="after")
    def enforce_route_limit(self) -> "TextProcessingResult":
        limit = ROUTE_ANSWER_LIMITS[self.category.value]
        if len(self.final_answer) > limit:
            raise ValueError(
                f"final_answer длиннее лимита маршрута {self.category.value}: {limit}"
            )
        return self
