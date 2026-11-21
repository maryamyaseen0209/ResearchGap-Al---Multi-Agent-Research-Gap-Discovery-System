"""Safe, bounded Phase 5 external literature-verification limits."""

from pydantic import BaseModel, Field, model_validator


class VerificationSettings(BaseModel):
    max_queries_per_gap: int = Field(default=2, ge=1, le=3)
    max_results_per_query: int = Field(default=5, ge=1, le=15)
    max_works_per_gap: int = Field(default=10, ge=1, le=30)
    max_abstract_characters: int = Field(default=2000, ge=300, le=8000)
    max_context_characters: int = Field(default=22000, ge=3000, le=60000)

    @model_validator(mode="after")
    def ensure_search_capacity(self):
        if self.max_works_per_gap < self.max_results_per_query:
            raise ValueError("max_works_per_gap must be >= max_results_per_query")
        return self
