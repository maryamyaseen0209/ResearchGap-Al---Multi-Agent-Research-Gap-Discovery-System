"""Phase 6 production LangGraph orchestration for existing Phases 1–5 agents.

Only the real `langgraph` package can execute the production graph. The graph
is built lazily so earlier phases still run even if the optional package has
not been installed yet. The offline unit tests use a *test-only* graph double.
"""

from __future__ import annotations

from uuid import uuid4

from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.comparison_schemas import ComparisonResult
from researchgap.core.gap_schemas import GapDiscoveryResult, ResearchGap
from researchgap.core.schemas import BatchResult
from researchgap.core.verification_schemas import LiteratureVerificationResult
from researchgap.core.workflow_schemas import WorkflowResult, WorkflowState
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.comparison_service import compare_batch
from researchgap.services.gap_service import discover_gaps
from researchgap.services.verification_service import verify_gaps


class WorkflowDependencyError(RuntimeError):
    """The production orchestration dependency was not installed."""


def _event(stage: str, status: str, message: str) -> list[dict]:
    return [{"stage": stage, "status": status, "message": message[:300]}]


def _failure_message(stage: str, exc: Exception) -> str:
    # No untrusted paper, model response, HTTP body, or API secret in UI logs.
    return f"{stage} could not finish ({type(exc).__name__}). Inspect input and service configuration."


def _fingerprint(gap: ResearchGap) -> tuple[str, tuple[str, ...]]:
    return gap.kind, tuple(sorted(f.fact_id for f in gap.cited_facts))


class ResearchWorkflow:
    """Run analyze -> compare -> discover -> verify -> [revise once] -> verify.

    No OpenAlex/LLM tools are called until the caller explicitly invokes `run`.
    The input is a previously extracted Phase 1 BatchResult; rejected, partial,
    and failed stages are preserved in the returned portable JSON result.
    """

    def __init__(
        self, analyzer: PaperAnalyzer, comparator: PaperComparator,
        discoverer: GapDiscoverer, verifier: GapVerifier,
        *, enable_revision: bool = True, checkpointer: bool = True,
    ):
        self.analyzer = analyzer
        self.comparator = comparator
        self.discoverer = discoverer
        self.verifier = verifier
        self.enable_revision = enable_revision
        self.graph = self._compile_graph(checkpointer)

    def _compile_graph(self, checkpoint: bool):
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError as exc:
            raise WorkflowDependencyError(
                "LangGraph is required for Phase 6. Run: pip install -r requirements.txt"
            ) from exc

        builder = StateGraph(WorkflowState)
        builder.add_node("analyze", self._analyze)
        builder.add_node("compare", self._compare)
        builder.add_node("discover", self._discover)
        builder.add_node("verify", self._verify)
        builder.add_node("revise", self._revise)
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "analyze")
        builder.add_conditional_edges(
            "analyze", self._after_analysis,
            {"compare": "compare", "finish": "finish"},
        )
        builder.add_conditional_edges(
            "compare", self._after_comparison,
            {"discover": "discover", "finish": "finish"},
        )
        builder.add_conditional_edges(
            "discover", self._after_discovery,
            {"verify": "verify", "finish": "finish"},
        )
        builder.add_conditional_edges(
            "verify", self._after_verification,
            {"revise": "revise", "finish": "finish"},
        )
        builder.add_conditional_edges(
            "revise", self._after_revision,
            {"verify": "verify", "finish": "finish"},
        )
        builder.add_edge("finish", END)
        return builder.compile(checkpointer=InMemorySaver() if checkpoint else None)

    def _analyze(self, state: WorkflowState) -> dict:
        try:
            batch = BatchResult.model_validate(state["batch"])
            if not batch.ready_for_analysis:
                return {"blocked_reason": "At least three readable research papers are required.",
                        "events": _event("analysis", "skipped", "Insufficient processed papers.")}
            result = analyze_batch(batch, self.analyzer)
            status = "ok" if not result.failures else "partial"
            return {"paper_analyses": result.model_dump(mode="json"),
                    "events": _event("analysis", status,
                                     f"Analyzed {result.successful_count}/{len(batch.papers)} papers.")}
        except (ValueError, TypeError, KeyError) as exc:
            return {"blocked_reason": _failure_message("Analysis", exc),
                    "events": _event("analysis", "failed", _failure_message("Analysis", exc))}

    @staticmethod
    def _after_analysis(state: WorkflowState) -> str:
        data = state.get("paper_analyses")
        return "compare" if data and len(data.get("analyses", [])) >= 3 else "finish"

    def _compare(self, state: WorkflowState) -> dict:
        try:
            previous = AnalysisBatchResult.model_validate(state["paper_analyses"])
            result = compare_batch(previous, self.comparator)
            return {"paper_comparison": result.model_dump(mode="json"),
                    "events": _event("comparison", "partial" if result.status != "complete" else "ok",
                                     f"Compared {result.paper_count} papers; {len(result.insights)} cross-paper insights.")}
        except (ValueError, TypeError, KeyError) as exc:
            return {"blocked_reason": _failure_message("Comparison", exc),
                    "events": _event("comparison", "failed", _failure_message("Comparison", exc))}

    @staticmethod
    def _after_comparison(state: WorkflowState) -> str:
        return "discover" if state.get("paper_comparison") else "finish"

    def _discover(self, state: WorkflowState) -> dict:
        try:
            comparison = ComparisonResult.model_validate(state["paper_comparison"])
            result = discover_gaps(comparison, self.discoverer)
            return {"research_gaps": result.model_dump(mode="json"),
                    "events": _event("discovery", "partial" if result.status != "complete" else "ok",
                                     f"Found {len(result.gaps)} evidence-linked candidate gaps.")}
        except (ValueError, TypeError, KeyError) as exc:
            return {"blocked_reason": _failure_message("Discovery", exc),
                    "events": _event("discovery", "failed", _failure_message("Discovery", exc))}

    @staticmethod
    def _after_discovery(state: WorkflowState) -> str:
        return "verify" if state.get("research_gaps", {}).get("gaps") else "finish"

    def _verify(self, state: WorkflowState) -> dict:
        try:
            discovery = GapDiscoveryResult.model_validate(state["research_gaps"])
            result = verify_gaps(discovery, self.verifier)
            patch = {"literature_verification": result.model_dump(mode="json"),
                     "events": _event("verification", "partial" if result.status != "complete" else "ok",
                                      f"Checked {len(result.verifications)} candidates; status={result.status}.")}
            if not state.get("revision_attempted"):
                patch["initial_literature_verification"] = result.model_dump(mode="json")
            return patch
        except (ValueError, TypeError, KeyError) as exc:
            return {"literature_verification": None,
                    "blocked_reason": _failure_message("Verification", exc),
                    "events": _event("verification", "failed", _failure_message("Verification", exc))}

    def _after_verification(self, state: WorkflowState) -> str:
        if (self.enable_revision and not state.get("blocked_reason")
                and not state.get("revision_attempted") and state.get("literature_verification")):
            verification = LiteratureVerificationResult.model_validate(state["literature_verification"])
            if any(item.status == "challenged" for item in verification.verifications):
                return "revise"
        return "finish"

    def _revise(self, state: WorkflowState) -> dict:
        """Replace challenged gaps only; retain unchallenged findings and evidence."""
        current = LiteratureVerificationResult.model_validate(state["literature_verification"])
        discovery = GapDiscoveryResult.model_validate(state["research_gaps"])
        comparison = ComparisonResult.model_validate(state["paper_comparison"])
        challenged = {item.gap_id for item in current.verifications if item.status == "challenged"}
        rejected = [g for g in discovery.gaps if g.gap_id in challenged]
        keep = [g for g in discovery.gaps if g.gap_id not in challenged]
        patch = {"revision_attempted": True, "revision_count": 1}
        try:
            suggestions = self.discoverer.discover(
                comparison, rejected_titles=[gap.title for gap in rejected],
            )
        except (ValueError, TypeError) as exc:
            patch.update({"revision_applied": False,
                          "events": _event("revision", "failed", _failure_message("Revision", exc))})
            return patch
        existing = {_fingerprint(g) for g in discovery.gaps}
        taken_titles = {g.title.strip().casefold() for g in discovery.gaps}
        replacements = [g for g in suggestions.gaps
                        if _fingerprint(g) not in existing
                        and g.title.strip().casefold() not in taken_titles]
        replacements = replacements[:max(0, 5 - len(keep))]
        if not replacements:
            patch.update({"revision_applied": False,
                          "events": _event("revision", "skipped",
                                           "No genuinely different, source-grounded replacement found; original verdicts retained.")})
            return patch
        new_gaps = [*keep, *replacements]
        # IDs may differ between generated batches; assign them consistently
        # before re-verification so each final assessment maps to its own gap.
        new_gaps = [gap.model_copy(update={"gap_id": f"gap_{i:02d}"})
                    for i, gap in enumerate(new_gaps, 1)]
        revised = discovery.model_copy(deep=True, update={
            "gaps": new_gaps, "status": "partial",
            "warnings": [*discovery.warnings,
                         "One bounded revision replaced challenged suggestions. All new gaps require re-verification."],
        })
        patch.update({"research_gaps": revised.model_dump(mode="json"),
                      "revision_applied": True,
                      "events": _event("revision", "partial",
                                       f"Retained {len(keep)} candidate gaps; proposed {len(replacements)} replacement(s).")})
        return patch

    @staticmethod
    def _after_revision(state: WorkflowState) -> str:
        return "verify" if state.get("revision_applied") else "finish"

    @staticmethod
    def _finish(state: WorkflowState) -> dict:
        if state.get("blocked_reason"):
            message = state["blocked_reason"]
            status = "failed"
        elif not state.get("paper_comparison"):
            message = "Stopped: at least three successfully analyzed papers are needed."
            status = "skipped"
        elif not state.get("research_gaps", {}).get("gaps"):
            message = "Stopped: insufficient grounded research gaps for literature verification."
            status = "skipped"
        else:
            message = "Workflow finished. Scientific novelty is not established."
            status = "ok"
        return {"events": _event("finish", status, message)}

    def run(self, batch: BatchResult, *, run_id: str | None = None) -> WorkflowResult:
        """Invoke the compiled graph once; every external call happens inside this method."""
        if not isinstance(batch, BatchResult):
            raise TypeError("run requires the Phase 1 BatchResult object")
        identifier = uuid4().hex if run_id is None else run_id
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("run_id must not be empty")
        state = self.graph.invoke(
            {"batch": batch.model_dump(mode="json"), "revision_count": 0,
             "revision_attempted": False, "revision_applied": False, "events": []},
            config={"configurable": {"thread_id": identifier}},
        )
        return self._to_result(state, identifier)

    def resume(self, run_id: str) -> WorkflowResult:
        """Resume an unfinished checkpoint in the *same runner instance*."""
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        state = self.graph.invoke(None, config={"configurable": {"thread_id": run_id}})
        if not state:
            raise ValueError("No checkpoint exists for this run ID")
        return self._to_result(state, run_id)

    @staticmethod
    def _to_result(state: dict, run_id: str) -> WorkflowResult:
        if state.get("blocked_reason") or not state.get("paper_comparison"):
            status = "stopped"
        elif (not state.get("literature_verification")
              or state["literature_verification"].get("status") != "complete"
              or state.get("revision_attempted") and not state.get("revision_applied")
              or state.get("research_gaps", {}).get("status") != "complete"
              or state.get("paper_comparison", {}).get("status") != "complete"
              or state.get("paper_analyses", {}).get("failures")):
            status = "partial"
        else:
            status = "complete"
        return WorkflowResult(
            run_id=run_id, status=status, blocked_reason=state.get("blocked_reason"),
            paper_analyses=state.get("paper_analyses"),
            paper_comparison=state.get("paper_comparison"),
            research_gaps=state.get("research_gaps"),
            literature_verification=state.get("literature_verification"),
            initial_literature_verification=state.get("initial_literature_verification"),
            revision_attempted=state.get("revision_attempted", False),
            revision_applied=state.get("revision_applied", False),
            revision_count=state.get("revision_count", 0),
            events=state.get("events", []),
        )
