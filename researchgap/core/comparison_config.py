"""Small, explicit LLM context limits for Phase 3 comparisons."""

from pydantic import BaseModel, Field


class ComparisonSettings(BaseModel):
    max_facts_per_category_per_paper: int = Field(default=4, ge=1, le=10)
    max_context_characters: int = Field(default=24_000, ge=1200, le=100_000)
    max_insights: int = Field(default=12, ge=1, le=30)
