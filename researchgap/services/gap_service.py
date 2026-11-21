"""Phase 4 discovery service and future LangGraph node adapter."""

from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.core.comparison_schemas import ComparisonResult
from researchgap.core.gap_schemas import GapDiscoveryResult


def discover_gaps(comparison: ComparisonResult, discoverer: GapDiscoverer) -> GapDiscoveryResult:
    """Discover candidate gaps only; do not perform Phase 5 literature verification."""
    return discoverer.discover(comparison)


class GapDiscoveryNode:
    """Callable state patch for Phase 6 unified LangGraph orchestration."""

    def __init__(self, discoverer: GapDiscoverer):
        self.discoverer = discoverer

    def __call__(self, state: dict) -> dict:
        comparison = state["paper_comparison"]
        if isinstance(comparison, dict):
            comparison = ComparisonResult.model_validate(comparison)
        result = discover_gaps(comparison, self.discoverer)
        return {"research_gaps": result.model_dump(mode="json")}
