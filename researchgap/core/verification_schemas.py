"""Phase 5: verification is a scoped scholarly search, NEVER proof of novelty."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchgap.core.gap_schemas import GapDiscoveryResult

GapVerificationStatus = Literal["challenged", "no_direct_match_found", "inconclusive"]
VerificationRunStatus = Literal["complete", "partial", "unavailable", "no_gaps"]
WorkRelevance = Literal["potentially_addresses", "related", "uncertain"]


class LiteratureWork(BaseModel):
    """A record actually returned by OpenAlex, with canonical OpenAlex link."""

    openalex_id: str = Field(pattern=r"^W[0-9]+$")
    title: str = Field(min_length=1, max_length=500)
    url: str
    publication_year: int | None = Field(default=None, ge=1800, le=2200)
    doi: str | None = None
    abstract: str | None = None
    cited_by_count: int | None = Field(default=None, ge=0)

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"https://openalex\.org/W[0-9]+", value):
            raise ValueError("OpenAlex paper link must be a canonical HTTPS work URL")
        return value


class WorkAssessment(BaseModel):
    """Model judgement whose source quote is validated against work metadata."""

    openalex_id: str
    relevance: WorkRelevance
    reasoning: str = Field(min_length=12, max_length=700)
    evidence_quote: str | None = Field(default=None, max_length=700)


class GapVerification(BaseModel):
    gap_id: str
    title: str
    status: GapVerificationStatus
    search_queries: list[str]
    searches_attempted: int = Field(ge=0)
    searches_succeeded: int = Field(ge=0)
    works_retrieved: int = Field(ge=0)
    works_reviewed: int = Field(ge=0)
    works: list[LiteratureWork] = Field(default_factory=list)
    assessments: list[WorkAssessment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    novelty_established: Literal[False] = False
    human_review_required: Literal[True] = True


class LiteratureVerificationResult(BaseModel):
    """Includes the complete original Phase 4 evidence for portable JSON export."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["OpenAlex"] = "OpenAlex"
    model: str
    status: VerificationRunStatus
    discovery_status: str
    gap_discovery: GapDiscoveryResult
    verifications: list[GapVerification]
    warnings: list[str] = Field(default_factory=list)
    novelty_established: Literal[False] = False
    human_review_required: Literal[True] = True
