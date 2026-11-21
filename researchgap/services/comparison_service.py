"""Phase 3 batch comparison service and a future LangGraph-compatible state node."""

from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.comparison_schemas import ComparisonResult


def compare_batch(analyses: AnalysisBatchResult, comparator: PaperComparator) -> ComparisonResult:
    """Compare only successfully analyzed papers; no automatic external calls."""
    return comparator.compare(analyses)


class PaperComparisonNode:
    """LangGraph-ready callable; full multi-agent graph is planned for Phase 6."""

    def __init__(self, comparator: PaperComparator):
        self.comparator = comparator

    def __call__(self, state: dict) -> dict:
        previous = state["paper_analyses"]
        if isinstance(previous, dict):
            previous = AnalysisBatchResult.model_validate(previous)
        result = compare_batch(previous, self.comparator)
        return {"paper_comparison": result.model_dump(mode="json")}
