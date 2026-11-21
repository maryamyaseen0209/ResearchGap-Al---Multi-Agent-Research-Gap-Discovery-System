"""Full Phase 1–8 offline smoke test using fictional PDFs and simulated APIs.

This validates REAL pdf parsing, agent business logic, citation mapping, report
writing and Phase 8 evaluation. When LangGraph is installed it also invokes
REAL LangGraph; otherwise it uses an explicitly labeled sequential orchestration
for this offline demonstration only (never a production fallback).

No fake results should be presented as experimental superiority or scientific
novelty. This is a SOFTWARE smoke test with fictional demonstration data.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

from examples.demo_phase4_offline import OfflineFourPhaseModel
from examples.demo_phase5_offline import OfflineReviewModel, OfflineOpenAlex
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.evaluation.baseline import SinglePassBaseline
from researchgap.evaluation.metrics import compare_systems, create_review_template
from researchgap.evaluation.telemetry import MeteredLLM
from researchgap.reports import generate_research_report
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.comparison_service import compare_batch
from researchgap.services.gap_service import discover_gaps
from researchgap.services.verification_service import verify_gaps
from researchgap.services.workflow_service import ResearchWorkflow


class OfflineBaselineModel:
    model = "fictional-offline-baseline"

    def extract_json(self, system, user):
        # Uses only supplied document facts. Synthetic deterministic response.
        facts = json.loads(user)["facts"]
        limitations = [f for f in facts if f["category"] == "limitation"]
        others = [f for f in facts if f["category"] == "finding"]
        if not limitations or not others:
            return {"gaps": []}
        return {"gaps": [
            {"title": "Explore limits of the provided educational case studies",
             "rationale": "The available abstracts leave conditions for generalization uncertain.",
             "research_question": "How do conclusions transfer to another classroom population?",
             "fact_ids": [limitations[0]["fact_id"], others[-1]["fact_id"]]},
            {"title": "Hypothetical investigation with invalid source attribution",
             "rationale": "This fictional suggestion intentionally includes an invented fact ID.",
             "research_question": "What evidence would be necessary to evaluate this topic?",
             "fact_ids": [limitations[0]["fact_id"], "not_a_real_fact_id"]},
        ]}


def execute(out_dir: Path, *, require_langgraph: bool = False) -> dict:
    sample = Path(__file__).resolve().parent / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])
    assert batch.ready_for_analysis, "Sample PDF processing failed"
    model = MeteredLLM(OfflineFourPhaseModel())
    analyzer = PaperAnalyzer(model)
    comparator = PaperComparator(model)
    discoverer = GapDiscoverer(model)
    verifier = GapVerifier(OfflineReviewModel(), OfflineOpenAlex())

    has_langgraph = importlib.util.find_spec("langgraph") is not None
    if require_langgraph and not has_langgraph:
        raise RuntimeError("Real LangGraph is not installed; cannot run --require-langgraph")

    start = perf_counter()
    if has_langgraph:
        workflow = ResearchWorkflow(analyzer, comparator, discoverer, verifier, enable_revision=False)
        result = workflow.run(batch, run_id="phase8-fictional-smoke")
        assert workflow.resume("phase8-fictional-smoke") == result
        graph_mode = "real_langgraph"
    else:
        # Deliberately not pretending to have executed LangGraph.
        analyses = analyze_batch(batch, analyzer)
        comparison = compare_batch(analyses, comparator)
        gaps = discover_gaps(comparison, discoverer)
        verified = verify_gaps(gaps, verifier)
        result = WorkflowResult(run_id="phase8-fictional-sequential-smoke", status="complete",
            paper_analyses=analyses, paper_comparison=comparison, research_gaps=gaps,
            literature_verification=verified, initial_literature_verification=verified)
        graph_mode = "sequential_offline_only_langgraph_not_installed"
    workflow_seconds = perf_counter() - start

    assert result.paper_analyses and len(result.paper_analyses.analyses) == 3
    assert result.paper_comparison and len(result.paper_comparison.comparison_table) == 3
    assert result.research_gaps and result.research_gaps.gaps
    assert result.literature_verification
    assert not result.novelty_established

    base_model = MeteredLLM(OfflineBaselineModel())
    baseline = SinglePassBaseline(base_model).run(result.paper_analyses)
    audit = compare_systems(batch, baseline, result)
    assert audit.multi_agent.gap_count >= 1
    assert audit.multi_agent.gaps_with_invalid_reference == 0
    assert audit.multi_agent.page_citation_integrity_rate == 1
    assert audit.baseline.gaps_with_invalid_reference == 1
    assert audit.multi_agent.human_support_rate is None
    report = generate_research_report(batch=batch, workflow=result,
                                      title="Offline Fictional Research Demonstration")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "processed_papers.json").write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "paper_analyses.json").write_text(result.paper_analyses.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "workflow_results.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "baseline_results.json").write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "evaluation_results.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "human_review_template.csv").write_text(create_review_template(baseline, result), encoding="utf-8")
    (out_dir / "research_report.md").write_text(report.markdown, encoding="utf-8")
    (out_dir / "research_report.html").write_text(report.html, encoding="utf-8")
    (out_dir / "research_report.json").write_text(report.json_text, encoding="utf-8")
    metrics = {"mode": graph_mode, "dataset": "3 fictional PDFs; NOT scientific evaluation",
               "workflow_wall_seconds": round(workflow_seconds, 4),
               "agents_llm_usage": model.measurements(),
               "baseline_llm_usage": base_model.measurements(),
               "live_groq_calls": 0, "live_openalex_calls": 0,
               "real_scientific_accuracy_evaluated": False}
    (out_dir / "runtime_measurements.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Phase 8 software-level end-to-end test")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_output/offline_smoke"))
    parser.add_argument("--require-langgraph", action="store_true")
    args = parser.parse_args(argv)
    stats = execute(args.output_dir, require_langgraph=args.require_langgraph)
    print("ResearchGap AI Phase 1–8 OFFLINE SOFTWARE SMOKE: PASS")
    print(f"Graph mode: {stats['mode']}")
    print(f"Outputs saved in: {args.output_dir}")
    print("This is fictional test data. No live model/scientific accuracy claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
