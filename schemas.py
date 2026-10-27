"""Structured result of the day-1 text processing step."""

from pydantic import BaseModel, Field, field_validator


class TextProcessingResult(BaseModel):
    """Summary, three key points, and a short helpful response."""

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=3, max_length=3)
    response: str = Field(min_length=1)

    @field_validator("summary", "response")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("key_points")
    @classmethod
    def strip_key_points(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("key points must not be blank")
        return cleaned
