"""Phase 4 bounded research-gap discovery context and output limits."""

from pydantic import BaseModel, Field


class GapSettings(BaseModel):
    max_facts_per_category_per_paper: int = Field(default=4, ge=1, le=10)
    max_context_characters: int = Field(default=24_000, ge=1200, le=100_000)
    max_candidate_gaps: int = Field(default=5, ge=1, le=5)
    max_comparison_insights: int = Field(default=8, ge=0, le=25)
