"""Phase 3 contracts for auditable, evidence-linked comparisons across papers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchgap.core.analysis_schemas import EvidenceReference, FactKind

ComparisonKind = Literal[
    "method_similarity", "method_difference", "dataset_comparison",
    "finding_comparison", "conflicting_findings", "shared_limitation",
    "objective_difference", "contribution_comparison",
]
ComparisonStatus = Literal["complete", "partial", "no_supported_comparisons"]


class CitedPaperFact(BaseModel):
    """A citation is derived from an existing Phase 2 verified fact, never from an LLM."""

    fact_id: str
    document_id: str
    filename: str
    category: FactKind
    statement: str
    evidence: EvidenceReference


class PaperComparisonRow(BaseModel):
    """A deterministic paper-by-paper table, including missing categories explicitly."""

    document_id: str
    filename: str
    title: str
    analysis_status: Literal["complete", "partial", "no_verified_facts"]
    objectives: list[CitedPaperFact] = Field(default_factory=list)
    methodologies: list[CitedPaperFact] = Field(default_factory=list)
    datasets: list[CitedPaperFact] = Field(default_factory=list)
    findings: list[CitedPaperFact] = Field(default_factory=list)
    limitations: list[CitedPaperFact] = Field(default_factory=list)
    contributions: list[CitedPaperFact] = Field(default_factory=list)


class ComparisonInsight(BaseModel):
    kind: ComparisonKind
    statement: str = Field(min_length=10)
    cited_facts: list[CitedPaperFact] = Field(min_length=2)


class ComparisonCoverage(BaseModel):
    total_source_facts: int = Field(ge=0)
    facts_sent_to_llm: int = Field(ge=0)
    input_characters: int = Field(ge=0)
    truncated: bool
    included_document_count: int = Field(ge=0)


class ComparisonResult(BaseModel):
    model: str
    status: ComparisonStatus
    paper_count: int = Field(ge=3, le=5)
    comparison_table: list[PaperComparisonRow] = Field(min_length=3, max_length=5)
    insights: list[ComparisonInsight] = Field(default_factory=list)
    coverage: ComparisonCoverage
    warnings: list[str] = Field(default_factory=list)


class CandidateInsight(BaseModel):
    """Untrusted model suggestion. Its cited IDs and paper categories must be checked."""

    model_config = ConfigDict(extra="ignore")
    kind: ComparisonKind
    statement: str = Field(min_length=10, max_length=1000)
    fact_ids: list[str] = Field(min_length=2, max_length=12)


class CandidateComparison(BaseModel):
    insights: list[CandidateInsight] = Field(default_factory=list, max_length=50)
