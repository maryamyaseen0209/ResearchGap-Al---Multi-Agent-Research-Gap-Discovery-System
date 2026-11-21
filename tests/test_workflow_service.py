"""Phase 6 controlled node/routing tests, NOT real LangGraph execution."""

import json

import pytest

from examples.demo_phase4_offline import OfflineFourPhaseModel
from examples.demo_phase5_offline import ABSTRACT, OfflineOpenAlex
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.workflow_service import ResearchWorkflow, WorkflowDependencyError
from langgraph_test_double import install


class ScientificFakeLLM(OfflineFourPhaseModel):
    def __init__(self, *, valid_quote=True, revise=False, no_gaps=False, failures=False):
        self.valid_quote = valid_quote
        self.revise = revise
        self.no_gaps = no_gaps
        self.failures = failures
        self.revision_requests = 0
        self.requests = []

    def extract_json(self, system, user):
        data = json.loads(user)
        self.requests.append(data)
        if "candidate_gap" in data and "works" in data:
            return {"assessments": [{
                "openalex_id": w["openalex_id"], "relevance": "potentially_addresses",
                "reasoning": "Fictional abstract appears to discuss the simulated problem.",
                "evidence_quote": ABSTRACT if self.valid_quote else "Entirely invented text not in this abstract",
            } for w in data["works"]]}
        if "comparison_insights" in data:
            if self.no_gaps:
                return {"gaps": []}
            if "revision_feedback" in data:
                self.revision_requests += 1
                if self.revise:
                    limits = [f["fact_id"] for f in data["facts"] if f["category"] == "limitation"]
                    findings = [f["fact_id"] for f in data["facts"] if f["category"] == "finding"]
                    return {"gaps": [{
                        "kind": "evaluation_opportunity",
                        "title": "Controlled evaluation of classroom model robustness",
                        "rationale": "Different paper limitations motivate validation of model robustness in a small study.",
                        "fact_ids": [limits[0], findings[1]],
                        "research_questions": [{"text": "How can classroom model robustness be tested in another population?"}],
                        "importance": "medium", "importance_reason": "It could improve confidence in findings.",
                        "feasibility": "medium", "feasibility_reason": "Requires limited resources and independent review.",
                    }]}
                return super().extract_json(system, user)
        return super().extract_json(system, user)


class EmptySearch:
    def __init__(self):
        self.calls = 0

    def search(self, query, limit=5):
        self.calls += 1
        return []


class ChallengeThenClear:
    def __init__(self):
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        if "robustness" in query:
            return []
        return OfflineOpenAlex().search(query, limit)


@pytest.fixture
def processed_batch():
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    return process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])


@pytest.fixture
def patch_graph(monkeypatch):
    install(monkeypatch)


def make_runner(model, searcher, **kwargs):
    return ResearchWorkflow(
        PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
        GapVerifier(model, searcher), **kwargs,
    )


def test_requires_real_langgraph_by_default(processed_batch, monkeypatch):
    # Simulate a missing dependency even when CI has installed real LangGraph.
    # No production fallback to a fake graph may ever be silently installed.
    import builtins
    original = builtins.__import__

    def missing_langgraph(name, *args, **kwargs):
        if name.startswith("langgraph"):
            raise ImportError("Forced missing LangGraph for negative dependency test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_langgraph)
    with pytest.raises(WorkflowDependencyError, match="LangGraph is required"):
        make_runner(ScientificFakeLLM(), EmptySearch())


def test_unified_success_without_revision(patch_graph, processed_batch):
    model = ScientificFakeLLM()
    searcher = EmptySearch()
    runner = make_runner(model, searcher)
    result = runner.run(processed_batch, run_id="test-ok")
    assert result.status == "complete"
    assert len(result.paper_analyses.analyses) == 3
    assert len(result.paper_comparison.comparison_table) == 3
    assert len(result.research_gaps.gaps) == 3
    assert result.literature_verification.status == "complete"
    assert all(v.status == "no_direct_match_found" for v in result.literature_verification.verifications)
    assert not result.revision_attempted
    assert [e.stage for e in result.events] == [
        "analysis", "comparison", "discovery", "verification", "finish",
    ]
    assert result.novelty_established is False
    assert result.human_review_required is True
    assert searcher.calls > 0
    assert runner.resume("test-ok") == result
    assert WorkflowResult.model_validate_json(result.model_dump_json()) == result
    for gap in result.research_gaps.gaps:
        for fact in gap.cited_facts:
            page = next(p for p in processed_batch.papers if p.document_id == fact.document_id).pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote


def test_one_revision_when_challenged_and_reverify(patch_graph, processed_batch):
    model = ScientificFakeLLM(revise=True)
    searcher = ChallengeThenClear()
    result = make_runner(model, searcher).run(processed_batch)
    assert result.revision_attempted and result.revision_applied
    assert result.revision_count == 1 and model.revision_requests == 1
    assert any(v.status == "challenged" for v in result.initial_literature_verification.verifications)
    assert len(result.research_gaps.gaps) == 1
    assert result.research_gaps.gaps[0].title == "Controlled evaluation of classroom model robustness"
    assert result.literature_verification.gap_discovery == result.research_gaps
    assert all(v.status == "no_direct_match_found" for v in result.literature_verification.verifications)
    assert [e.stage for e in result.events].count("revision") == 1
    assert [e.stage for e in result.events].count("verification") == 2
    assert len(result.events) <= 7
    assert not result.novelty_established


def test_one_revision_no_replacement_keeps_challenged(patch_graph, processed_batch):
    model = ScientificFakeLLM(revise=False)
    result = make_runner(model, OfflineOpenAlex()).run(processed_batch)
    assert result.revision_attempted is True
    assert result.revision_applied is False
    assert result.revision_count == 1
    assert result.status == "partial"
    assert result.literature_verification == result.initial_literature_verification
    assert all(item.status == "challenged" for item in result.literature_verification.verifications)
    assert sum(e.stage == "verification" for e in result.events) == 1
    assert sum(e.stage == "revision" for e in result.events) == 1


def test_disable_revision(patch_graph, processed_batch):
    result = make_runner(ScientificFakeLLM(), OfflineOpenAlex(), enable_revision=False).run(processed_batch)
    assert not result.revision_attempted
    assert result.revision_count == 0
    assert all(v.status == "challenged" for v in result.literature_verification.verifications)


def test_challenge_without_valid_quote_does_not_trigger_revision(patch_graph, processed_batch):
    result = make_runner(ScientificFakeLLM(valid_quote=False), OfflineOpenAlex()).run(processed_batch)
    assert not result.revision_attempted
    assert all(v.status != "challenged" for v in result.literature_verification.verifications)


def test_stops_cleanly_with_only_two_documents(patch_graph, processed_batch):
    batch = processed_batch.model_copy(deep=True)
    batch.papers.pop()
    search = EmptySearch()
    model = ScientificFakeLLM()
    result = make_runner(model, search).run(batch)
    assert result.status == "stopped"
    assert result.paper_analyses is None
    assert result.paper_comparison is None
    assert not search.calls and not model.requests
    assert [event.stage for event in result.events] == ["analysis", "finish"]


def test_stops_when_not_enough_papers_analyzed(patch_graph, processed_batch):
    class MixedAnalyzer:
        def analyze(self, paper):
            if paper.filename.endswith("sample_student_feedback.pdf"):
                raise ValueError("Simulated bad document")
            return PaperAnalyzer(ScientificFakeLLM()).analyze(paper)

    runner = make_runner(ScientificFakeLLM(), EmptySearch())
    runner.analyzer = MixedAnalyzer()
    result = runner.run(processed_batch)
    assert result.status == "stopped"
    assert result.paper_analyses.successful_count == 2
    assert len(result.paper_analyses.failures) == 1
    assert result.paper_comparison is None
    assert not result.literature_verification


def test_no_gaps_skips_verifier(patch_graph, processed_batch):
    searcher = EmptySearch()
    result = make_runner(ScientificFakeLLM(no_gaps=True), searcher).run(processed_batch)
    assert result.status == "partial"
    assert result.research_gaps.status == "no_grounded_gaps"
    assert result.literature_verification is None
    assert not searcher.calls
    assert result.events[-1].stage == "finish"


def test_partial_search_never_claims_novelty(patch_graph, processed_batch):
    from researchgap.integrations.openalex_client import LiteratureSearchError

    class FailingSearcher:
        def search(self, query, limit=5):
            raise LiteratureSearchError("NETWORK_ERROR", "Fake offline connection failure")

    result = make_runner(ScientificFakeLLM(), FailingSearcher()).run(processed_batch)
    assert result.status == "partial"
    assert result.literature_verification.status == "unavailable"
    assert not result.revision_attempted
    assert all(v.status == "inconclusive" for v in result.literature_verification.verifications)
    assert result.novelty_established is False


def test_node_errors_do_not_invoke_downstream(patch_graph, processed_batch):
    runner = make_runner(ScientificFakeLLM(), EmptySearch())
    class InvalidComparator:
        def compare(self, analyses):
            raise ValueError("Invalid source evidence")
    runner.comparator = InvalidComparator()
    result = runner.run(processed_batch)
    assert result.status == "stopped"
    assert result.blocked_reason.startswith("Comparison could not finish")
    assert result.paper_comparison is None
    assert [e.stage for e in result.events] == ["analysis", "comparison", "finish"]


def test_bad_run_id_rejected(patch_graph, processed_batch):
    runner = make_runner(ScientificFakeLLM(), EmptySearch())
    with pytest.raises(ValueError, match="run_id"):
        runner.run(processed_batch, run_id="")
    with pytest.raises(TypeError):
        runner.run({"papers": []})
    with pytest.raises(ValueError):
        runner.resume("")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.resume("unknown")
