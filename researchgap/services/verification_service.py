"""Phase 5 service + state-patch adapter for Phase 6 LangGraph."""

from researchgap.agents.gap_verifier import GapVerifier
from researchgap.core.gap_schemas import GapDiscoveryResult
from researchgap.core.verification_schemas import LiteratureVerificationResult


def verify_gaps(discovery: GapDiscoveryResult, verifier: GapVerifier) -> LiteratureVerificationResult:
    return verifier.verify(discovery)


class GapVerificationNode:
    def __init__(self, verifier: GapVerifier):
        self.verifier = verifier

    def __call__(self, state: dict) -> dict:
        discovery = state["research_gaps"]
        if isinstance(discovery, dict):
            discovery = GapDiscoveryResult.model_validate(discovery)
        result = verify_gaps(discovery, self.verifier)
        return {"literature_verification": result.model_dump(mode="json")}
