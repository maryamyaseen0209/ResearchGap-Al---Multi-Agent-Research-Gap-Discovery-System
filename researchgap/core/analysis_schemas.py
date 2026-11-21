"""Phase 2 structured, auditable research-paper analysis contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FactKind = Literal[
    "research_objective", "methodology", "dataset", "finding", "limitation", "contribution"
]

SupportStatus = Literal["supported", "partially_supported", "unsupported", "requires_human_review"]


class EvidenceReference(BaseModel):
    """Verified quote and exact location in the extracted PDF page."""

    document_id: str
    filename: str
    chunk_id: str
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=12)
    support_status: SupportStatus = "supported"


class GroundedFact(BaseModel):
    category: FactKind
    statement: str = Field(min_length=5)
    evidence: EvidenceReference
    support_status: SupportStatus = "supported"
    support_reason: str | None = None
    provider: str = "default"
    model: str = "default"
    limitation_type: str | None = None
    additional_evidence: list[EvidenceReference] = Field(default_factory=list)


class PaperCoverage(BaseModel):
    total_chunks: int = Field(ge=0)
    analyzed_chunks: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    analyzed_characters: int = Field(ge=0)
    truncated: bool
    batches_attempted: int = Field(ge=0)
    batches_succeeded: int = Field(ge=0)

    # Detailed coverage, batch accounting, and provenance tracking
    selected_chunks: int = 0
    attempted_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    pending_chunks: int = 0
    cached_chunks: int = 0
    total_planned_batches: int = 0
    pending_batches: int = 0
    attempted_batches: int = 0
    succeeded_batches: int = 0
    failed_batches: int = 0
    unique_analyzed_characters: int = 0
    batches_failed: int = 0
    batches_cached: int = 0
    selected_coverage_pct: float = 0.0
    document_coverage_pct: float = 0.0
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    document_id: str
    filename: str
    title: str
    title_source: Literal["pdf_metadata", "first_page", "filename"]
    model: str
    status: Literal["complete", "completed", "partial", "failed", "no_verified_facts", "pending"] = "complete"
    research_objectives: list[GroundedFact] = Field(default_factory=list)
    methodologies: list[GroundedFact] = Field(default_factory=list)
    datasets: list[GroundedFact] = Field(default_factory=list)
    findings: list[GroundedFact] = Field(default_factory=list)
    limitations: list[GroundedFact] = Field(default_factory=list)
    contributions: list[GroundedFact] = Field(default_factory=list)
    coverage: PaperCoverage
    warnings: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def verified_fact_count(self) -> int:
        return sum(
            len(getattr(self, name))
            for name in (
                "research_objectives", "methodologies", "datasets", "findings",
                "limitations", "contributions",
            )
        )

    @property
    def supported_fact_count(self) -> int:
        return sum(
            1 for name in (
                "research_objectives", "methodologies", "datasets", "findings",
                "limitations", "contributions",
            )
            for f in getattr(self, name)
            if f.support_status == "supported"
        )

    @property
    def review_required_fact_count(self) -> int:
        return sum(
            1 for name in (
                "research_objectives", "methodologies", "datasets", "findings",
                "limitations", "contributions",
            )
            for f in getattr(self, name)
            if f.support_status == "requires_human_review"
        )

    @property
    def is_failed(self) -> bool:
        return (
            (self.coverage.batches_attempted > 0 and self.coverage.batches_succeeded == 0)
            or self.status == "failed"
        )

    @property
    def is_partial(self) -> bool:
        return (
            self.coverage.batches_succeeded > 0
            and (
                self.coverage.batches_succeeded < self.coverage.batches_attempted
                or self.status == "partial"
                or self.coverage.truncated
            )
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.coverage.batches_succeeded == self.coverage.batches_attempted
            and self.coverage.batches_succeeded > 0
            and not self.coverage.truncated
            and self.status in ("complete", "completed")
        )


class AnalysisFailure(BaseModel):
    document_id: str
    filename: str
    code: str
    message: str


class AnalysisBatchResult(BaseModel):
    requested_count: int = Field(ge=0)
    analyses: list[PaperAnalysis] = Field(default_factory=list)
    failures: list[AnalysisFailure] = Field(default_factory=list)

    @property
    def completed_analyses(self) -> list[PaperAnalysis]:
        return [a for a in self.analyses if a.is_complete]

    @property
    def partial_analyses(self) -> list[PaperAnalysis]:
        return [a for a in self.analyses if a.is_partial]

    @property
    def failed_analyses(self) -> list[PaperAnalysis]:
        return [a for a in self.analyses if a.is_failed]

    @property
    def completed_count(self) -> int:
        return len(self.completed_analyses)

    @property
    def partial_count(self) -> int:
        return len(self.partial_analyses)

    @property
    def failed_count(self) -> int:
        failed_ids = {a.document_id for a in self.analyses if a.is_failed}
        failed_ids.update(getattr(f, "document_id", None) or f.filename for f in self.failures)
        return len(failed_ids)

    @property
    def successful_count(self) -> int:
        """Count of papers that produced at least one usable analysis batch."""
        return sum(1 for a in self.analyses if a.coverage.batches_succeeded > 0)

    @property
    def total_accepted_facts(self) -> int:
        return sum(a.verified_fact_count for a in self.analyses)

    @property
    def total_batches_succeeded(self) -> int:
        return sum(a.coverage.batches_succeeded for a in self.analyses)

    @property
    def total_batches_attempted(self) -> int:
        return sum(a.coverage.batches_attempted for a in self.analyses)

    @property
    def total_batches_planned(self) -> int:
        return sum(
            a.coverage.total_planned_batches or a.coverage.batches_attempted
            for a in self.analyses
        )


class CandidateFact(BaseModel):
    """Untrusted model output; it becomes GroundedFact only after source verification."""

    model_config = ConfigDict(extra="ignore")
    field: FactKind
    text: str = Field(min_length=5, max_length=1000)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=5, max_length=1500)


class CandidateExtraction(BaseModel):
    facts: list[CandidateFact] = Field(default_factory=list)
