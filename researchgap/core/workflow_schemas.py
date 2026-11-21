"""Phase 6 shared workflow state, source-preserving result and event contracts."""

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.comparison_schemas import ComparisonResult
from researchgap.core.gap_schemas import GapDiscoveryResult
from researchgap.core.verification_schemas import LiteratureVerificationResult


Stage = Literal["analysis", "comparison", "discovery", "verification", "revision", "finish"]
RunStatus = Literal["complete", "partial", "stopped"]


class WorkflowEvent(BaseModel):
    stage: Stage
    status: Literal["ok", "partial", "skipped", "failed"]
    message: str = Field(min_length=1, max_length=300)


class WorkflowState(TypedDict, total=False):
    """LangGraph state: each node writes patches, not entire mutable objects."""

    batch: dict
    paper_analyses: dict
    paper_comparison: dict
    research_gaps: dict
    literature_verification: dict
    initial_literature_verification: dict
    revision_count: int
    revision_applied: bool
    revision_attempted: bool
    events: Annotated[list[dict], add]
    blocked_reason: str


class WorkflowResult(BaseModel):
    """Portable Phase 6 report containing all successful, source-linked stages."""

    run_id: str
    status: RunStatus
    blocked_reason: str | None = None
    paper_analyses: AnalysisBatchResult | None = None
    paper_comparison: ComparisonResult | None = None
    research_gaps: GapDiscoveryResult | None = None
    literature_verification: LiteratureVerificationResult | None = None
    initial_literature_verification: LiteratureVerificationResult | None = None
    revision_attempted: bool = False
    revision_applied: bool = False
    revision_count: int = Field(default=0, ge=0, le=1)
    events: list[WorkflowEvent] = Field(default_factory=list)
    novelty_established: Literal[False] = False
    human_review_required: Literal[True] = True
