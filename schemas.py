"""Structured result of the text processing step."""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from prompts import (
    KEY_POINT_COUNT,
    MAX_KEY_POINT_CHARS,
    MAX_RESPONSE_CHARS,
    MAX_SUMMARY_CHARS,
)

KeyPoint = Annotated[str, Field(min_length=1, max_length=MAX_KEY_POINT_CHARS)]


class TextProcessingResult(BaseModel):
    """Summary, three key points, and a short helpful response."""

    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    key_points: list[KeyPoint] = Field(
        min_length=KEY_POINT_COUNT,
        max_length=KEY_POINT_COUNT,
    )
    response: str = Field(min_length=1, max_length=MAX_RESPONSE_CHARS)

    @field_validator("summary", "response", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
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
