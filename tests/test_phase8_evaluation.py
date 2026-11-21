"""Phase 8 software evaluation checks with fictional inputs only."""

import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from examples.demo_phase8_offline import OfflineBaselineModel, execute
from researchgap.evaluation.baseline import SinglePassBaseline, BaselineResult
from researchgap.evaluation.metrics import (
    compare_systems, create_review_template, read_review_labels,
)
from researchgap.evaluation.telemetry import MeteredLLM
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.schemas import BatchResult
from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.evaluation.cli import main as evaluation_cli


@pytest.fixture(scope="module")
def fictional_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("phase8")
    stats = execute(out)
    return out, stats


def test_full_offline_pipeline_and_report_integrity(fictional_outputs):
    folder, stats = fictional_outputs
    assert stats["live_groq_calls"] == 0
    assert stats["live_openalex_calls"] == 0
    assert stats["real_scientific_accuracy_evaluated"] is False
    assert stats["mode"] in ("real_langgraph", "sequential_offline_only_langgraph_not_installed")
    source = BatchResult.model_validate_json((folder / "processed_papers.json").read_text(encoding="utf-8"))
    workflow = WorkflowResult.model_validate_json((folder / "workflow_results.json").read_text(encoding="utf-8"))
    base = BaselineResult.model_validate_json((folder / "baseline_results.json").read_text(encoding="utf-8"))
    result = compare_systems(source, base, workflow)
    assert len(source.papers) == 3
    assert workflow.literature_verification
    assert workflow.research_gaps
    assert result.multi_agent.page_citation_integrity_rate == 1.0
    assert result.multi_agent.gaps_with_invalid_reference == 0
    assert result.baseline.gaps_with_invalid_reference == 1
    assert result.human_labels_total == 0
    assert result.multi_agent.human_support_rate is None
    assert result.baseline.human_unsupported_rate is None
    assert "<html" in (folder / "research_report.html").read_text(encoding="utf-8").lower()
    assert "Source:" in (folder / "research_report.md").read_text(encoding="utf-8")
    data = json.loads((folder / "research_report.json").read_text(encoding="utf-8"))
    assert data["report_metadata"]["novelty_established"] is False


def test_baseline_real_citations_and_hallucinated_ids_are_visible(fictional_outputs):
    folder, _ = fictional_outputs
    src = AnalysisBatchResult.model_validate_json((folder / "paper_analyses.json").read_text(encoding="utf-8"))
    result = SinglePassBaseline(OfflineBaselineModel()).run(src)
    assert result.model == "fictional-offline-baseline"
    assert result.status == "complete"
    assert len(result.gaps) == 2
    assert not result.gaps[0].unresolved_fact_ids
    assert result.gaps[1].unresolved_fact_ids == ["not_a_real_fact_id"]
    assert result.gaps[1].novelty_established is False
    assert 0 < result.facts_in_prompt <= result.total_input_facts


def test_baseline_rejects_few_papers_and_bad_model_output(fictional_outputs):
    folder, _ = fictional_outputs
    src = AnalysisBatchResult.model_validate_json((folder / "paper_analyses.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="3–5"):
        SinglePassBaseline(OfflineBaselineModel()).run(src.model_copy(update={"analyses": src.analyses[:2]}))

    class BadLLM:
        model = "bad"
        def extract_json(self, *_): return {"invalid": "not an array"}

    result = SinglePassBaseline(BadLLM()).run(src)
    assert result.status == "failed"
    assert not result.gaps


def test_human_csv_template_and_labels_do_not_forge_results(fictional_outputs):
    folder, _ = fictional_outputs
    source = BatchResult.model_validate_json((folder / "processed_papers.json").read_text(encoding="utf-8"))
    base = BaselineResult.model_validate_json((folder / "baseline_results.json").read_text(encoding="utf-8"))
    workflow = WorkflowResult.model_validate_json((folder / "workflow_results.json").read_text(encoding="utf-8"))
    template = create_review_template(base, workflow)
    rows = list(csv.DictReader(StringIO(template)))
    assert len(rows) == len(base.gaps) + len(workflow.research_gaps.gaps)
    assert all(not r["supported_by_evidence"] for r in rows)
    assert read_review_labels(template) == {}
    with pytest.raises(ValueError):
        read_review_labels("system,gap_id,supported_by_evidence\nbaseline,a,maybe\n")
    with pytest.raises(ValueError):
        read_review_labels("system,gap_id,supported_by_evidence\nbaseline,a,yes\nbaseline,a,no\n")
    selected = ("baseline", base.gaps[0].gap_id)
    audit = compare_systems(source, base, workflow, labels={selected: False})
    assert audit.baseline.human_reviewed_gaps == 1
    assert audit.baseline.human_unsupported_rate == 1
    assert audit.multi_agent.human_support_rate is None


def test_evaluation_cli_files_and_untrusted_review_ids(fictional_outputs, tmp_path):
    folder, _ = fictional_outputs
    out = tmp_path / "outputs"
    args = ["--batch", str(folder / "processed_papers.json"), "--baseline", str(folder / "baseline_results.json"),
            "--workflow", str(folder / "workflow_results.json"), "--output-dir", str(out)]
    assert evaluation_cli(args) == 0
    assert (out / "evaluation_results.json").is_file()
    assert (out / "evaluation_summary.md").is_file()
    assert (out / "human_review_template.csv").is_file()
    assert "Not measured" in (out / "evaluation_summary.md").read_text(encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_text("system,gap_id,supported_by_evidence\nbaseline,not-a-real-gap,yes\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluation_cli([*args, "--labels", str(bad)])


def test_metered_llm_character_counts_with_no_invented_tokens():
    meter = MeteredLLM(OfflineBaselineModel())
    result = meter.extract_json("system", json.dumps({"facts": []}))
    assert result == {"gaps": []}
    measurements = meter.measurements(input_usd_per_million=5, output_usd_per_million=10)
    assert measurements["calls"] == 1
    assert measurements["observed_input_tokens"] is None
    assert measurements["estimated_cost_usd"] is None
    assert measurements["llm_latency_seconds"] >= 0


def test_metered_llm_records_provided_token_counts_only():
    class Dummy:
        model = "tokens"
        observed_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        def extract_json(self, *_):
            self.observed_usage["prompt_tokens"] += 100
            self.observed_usage["completion_tokens"] += 20
            return {"ok": True}
    meter = MeteredLLM(Dummy())
    meter.extract_json("a", "b")
    meter.extract_json("a", "b")
    usage = meter.measurements(input_usd_per_million=1, output_usd_per_million=2)
    assert usage["observed_input_tokens"] == 200
    assert usage["observed_output_tokens"] == 40
    assert usage["estimated_cost_usd"] == pytest.approx(.00028)


def test_missing_langgraph_fails_loudly_when_requested(tmp_path):
    import importlib.util
    if importlib.util.find_spec("langgraph"):
        pytest.skip("Real LangGraph installed, handled by contract test")
    with pytest.raises(RuntimeError, match="Real LangGraph is not installed"):
        execute(tmp_path, require_langgraph=True)


def test_unicode_and_smart_quotes_cross_platform_encoding(tmp_path):
    """Regression test for Windows charmap/cp1252 UnicodeDecodeError on byte 0x9d/Unicode text."""
    test_file = tmp_path / "unicode_test.json"
    unicode_payload = {
        "title": "Étude sur l'apprentissage profond — “modèles multilingues” & α-β testing",
        "description": "Temperature range: -10°C ≤ T ≤ 45°C; precision: ±0.05 μm; Müller & François",
        "quotes": ["“Exact smart quote excerpt”", "‘Single smart quote’", "Japanese: 研究ギャップ", "Arabic: فجوة البحث"],
    }
    encoded_json = json.dumps(unicode_payload, ensure_ascii=False, indent=2)
    test_file.write_text(encoded_json, encoding="utf-8")
    loaded = json.loads(test_file.read_text(encoding="utf-8"))
    assert loaded["title"] == unicode_payload["title"]
    assert "“modèles multilingues”" in loaded["title"]
    assert "±0.05 μm" in loaded["description"]
    assert loaded["quotes"][2] == "Japanese: 研究ギャップ"
