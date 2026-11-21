"""Phase 4 evidence-grounded *candidate* gap discovery data contracts.

A valid citation is proof of source attribution, NOT proof of a gap's novelty.
External literature novelty checks are explicitly reserved for Phase 5.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchgap.core.comparison_schemas import CitedPaperFact, ComparisonStatus

GapKind = Literal[
    "shared_limitation", "dataset_coverage", "methodological_extension",
    "evaluation_opportunity",
]
GapStatus = Literal["complete", "partial", "no_grounded_gaps"]
Priority = Literal["low", "medium", "high"]


class ResearchQuestion(BaseModel):
    text: str = Field(min_length=15, max_length=300)

    @field_validator("text")
    @classmethod
    def require_question(cls, value: str) -> str:
        text = value.strip()
        if not text.endswith("?"):
            raise ValueError("Research questions must end with a question mark")
        return text


class CandidateGap(BaseModel):
    """Untrusted model output; validated against selected existing PDF evidence."""

    model_config = ConfigDict(extra="ignore")
    kind: GapKind
    title: str = Field(min_length=10, max_length=180)
    rationale: str = Field(min_length=25, max_length=1500)
    fact_ids: list[str] = Field(min_length=2, max_length=10)
    research_questions: list[ResearchQuestion] = Field(min_length=1, max_length=2)
    importance: Priority
    importance_reason: str = Field(min_length=15, max_length=500)
    feasibility: Priority
    feasibility_reason: str = Field(min_length=15, max_length=500)


class ResearchGap(BaseModel):
    gap_id: str
    kind: GapKind
    title: str
    rationale: str
    cited_facts: list[CitedPaperFact] = Field(min_length=2)
    research_questions: list[ResearchQuestion] = Field(min_length=1, max_length=2)
    importance: Priority
    importance_reason: str
    feasibility: Priority
    feasibility_reason: str
    literature_verification: Literal["not_performed"] = "not_performed"
    novelty_established: Literal[False] = False


class GapCoverage(BaseModel):
    total_source_facts: int = Field(ge=0)
    facts_sent_to_llm: int = Field(ge=0)
    input_characters: int = Field(ge=0)
    truncated: bool
    included_document_count: int = Field(ge=0)
    eligible_comparison_insights_used: int = Field(ge=0)


class GapDiscoveryResult(BaseModel):
    model: str
    status: GapStatus
    comparison_status: ComparisonStatus
    paper_count: int = Field(ge=3, le=5)
    gaps: list[ResearchGap] = Field(default_factory=list, max_length=5)
    coverage: GapCoverage
    warnings: list[str] = Field(default_factory=list)
    literature_verification: Literal["not_performed"] = "not_performed"
