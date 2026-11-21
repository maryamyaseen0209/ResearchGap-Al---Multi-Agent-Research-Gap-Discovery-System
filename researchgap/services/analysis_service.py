"""Analyze a processed PDF batch, preserving successful papers on per-paper errors."""

from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.core.analysis_schemas import AnalysisBatchResult, AnalysisFailure
from researchgap.core.schemas import BatchResult
from researchgap.integrations.groq_client import LLMError


def analyze_batch(batch: BatchResult, analyzer: PaperAnalyzer) -> AnalysisBatchResult:
    result = AnalysisBatchResult(requested_count=len(batch.papers))
    for paper in batch.papers:
        try:
            analysis = analyzer.analyze(paper)
            result.analyses.append(analysis)
            if analysis.is_failed:
                code = "RATE_LIMITED" if any("RATE_LIMITED" in w or "QUOTA" in w for w in analysis.warnings) else "ANALYSIS_FAILED"
                result.failures.append(
                    AnalysisFailure(
                        document_id=paper.document_id,
                        filename=paper.filename,
                        code=code,
                        message="Zero AI extraction batches succeeded for this paper.",
                    )
                )
        except (LLMError, ValueError) as exc:
            result.failures.append(
                AnalysisFailure(
                    document_id=paper.document_id,
                    filename=paper.filename,
                    code=exc.code if isinstance(exc, LLMError) else "ANALYSIS_ERROR",
                    message=str(exc),
                )
            )
    return result


class PaperAnalyzerNode:
    """Callable LangGraph-ready node: `node({\"batch\": BatchResult}) -> state patch`.

    Full multi-agent StateGraph integration remains Phase 6.
    """

    def __init__(self, analyzer: PaperAnalyzer):
        self.analyzer = analyzer

    def __call__(self, state: dict) -> dict:
        batch = state["batch"]
        if isinstance(batch, dict):
            batch = BatchResult.model_validate(batch)
        return {"paper_analyses": analyze_batch(batch, self.analyzer).model_dump(mode="json")}
