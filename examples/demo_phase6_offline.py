"""Real LangGraph graph, FICTIONAL papers/LLM/OpenAlex, zero external API calls.

Install `requirements-dev.txt` first. This confirms the actual LangGraph
package invokes our four agents and stores in-memory checkpoints, without
any Groq/OpenAlex credentials or external research accuracy claims.
"""

from pathlib import Path

from examples.demo_phase4_offline import OfflineFourPhaseModel
from examples.demo_phase5_offline import OfflineOpenAlex, OfflineReviewModel
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.workflow_service import ResearchWorkflow


def run():
    source = Path(__file__).resolve().parent / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(source.glob("*.pdf"))])
    paper_model = OfflineFourPhaseModel()
    workflow = ResearchWorkflow(
        PaperAnalyzer(paper_model), PaperComparator(paper_model),
        GapDiscoverer(paper_model), GapVerifier(OfflineReviewModel(), OfflineOpenAlex()),
        enable_revision=False,
    )
    result = workflow.run(batch, run_id="demo-phase6-offline")
    assert result.literature_verification is not None
    assert result.literature_verification.gap_discovery == result.research_gaps
    assert all(g.novelty_established is False for g in result.research_gaps.gaps)
    for gap in result.research_gaps.gaps:
        for fact in gap.cited_facts:
            paper = next(p for p in batch.papers if p.document_id == fact.document_id)
            page = paper.pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    assert workflow.resume("demo-phase6-offline") == result
    return result


def main():
    result = run()
    print("REAL LANGGRAPH, OFFLINE FICTIONAL DATA: this is NOT a scientific evaluation")
    for event in result.events:
        print(f" - {event.stage}: {event.status} — {event.message}")
    print(f"Workflow status: {result.status}; revisions: {result.revision_count}")
    print("Every candidate gap keeps source-PDF page citations; scientific novelty not established.")


if __name__ == "__main__":
    main()
