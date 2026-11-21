"""Real dependency integration check; skipped when LangGraph isn't installable."""

import importlib.util
import pytest

from examples.demo_phase4_offline import OfflineFourPhaseModel
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.workflow_service import ResearchWorkflow


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None,
                    reason="LangGraph is not installed in the testing environment")
def test_actual_langgraph_invocation_and_inmemory_checkpoint():
    from pathlib import Path
    from examples.demo_phase5_offline import OfflineOpenAlex
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])
    model = OfflineFourPhaseModel()
    runner = ResearchWorkflow(PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
                              GapVerifier(model, OfflineOpenAlex()), enable_revision=False)
    # Real workflow graph rather than an emulated routing engine.
    result = runner.run(batch, run_id="real-inmemory-checkpoint")
    assert result.literature_verification is not None
    assert len(result.events) == 5
    assert runner.resume("real-inmemory-checkpoint") == result
    # Confirm evidence citations resolve to original PDF text
    for gap in result.research_gaps.gaps:
        for fact in gap.cited_facts:
            paper = next(p for p in batch.papers if p.document_id == fact.document_id)
            page = paper.pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None,
                    reason="LangGraph is not installed in the testing environment")
def test_actual_langgraph_conditional_routing_and_bounded_revision():
    from pathlib import Path
    from test_workflow_service import ChallengeThenClear, ScientificFakeLLM
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])
    model = ScientificFakeLLM(revise=True)
    runner = ResearchWorkflow(PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
                              GapVerifier(model, ChallengeThenClear()), enable_revision=True)
    result = runner.run(batch, run_id="real-langgraph-revision")
    assert result.revision_attempted is True
    assert result.revision_applied is True
    assert result.revision_count == 1
    assert result.initial_literature_verification is not None
    assert any(item.status == "challenged" for item in result.initial_literature_verification.verifications)
    assert result.literature_verification is not None
    assert len(result.research_gaps.gaps) == 1
    assert result.research_gaps.gaps[0].title == "Controlled evaluation of classroom model robustness"
    assert len(result.events) == 7  # analyze, compare, discover, verify, revise, verify, finish
    assert runner.resume("real-langgraph-revision") == result


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None,
                    reason="LangGraph is not installed in the testing environment")
def test_actual_langgraph_handles_unexpected_error_safely():
    from pathlib import Path
    from examples.demo_phase5_offline import OfflineOpenAlex
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])

    class CrashingAnalyzerLLM(OfflineFourPhaseModel):
        def extract_json(self, system_message: str, user_message: str) -> dict:
            if "chunks" in user_message:
                raise KeyError("Simulated unhandled key error inside analyze node")
            return super().extract_json(system_message, user_message)

    model = CrashingAnalyzerLLM()
    runner = ResearchWorkflow(PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
                              GapVerifier(model, OfflineOpenAlex()), enable_revision=False)
    result = runner.run(batch, run_id="real-langgraph-error-safety")
    assert result.status == "stopped"
    assert result.blocked_reason is not None
    assert "Analysis could not finish" in result.blocked_reason
    assert any(e.stage == "analysis" and e.status == "failed" for e in result.events)
