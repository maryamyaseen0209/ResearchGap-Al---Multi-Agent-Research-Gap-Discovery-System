"""Phase 4 agent: source attribution, research questions and failure modes.

Scientific usefulness and originality are NOT established by these fake-model tests.
"""

import json

import pytest

from researchgap.agents.gap_discoverer import (
    SYSTEM_PROMPT, GapDiscoverer, resolve_candidate, select_gap_context,
    validated_table_facts,
)
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.gap_config import GapSettings
from researchgap.core.gap_schemas import GapDiscoveryResult, ResearchQuestion
from researchgap.integrations.groq_client import LLMError
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.comparison_service import compare_batch
from researchgap.services.gap_service import GapDiscoveryNode, discover_gaps


class FakeEvidenceExtractor:
    model = "phase2-fake"

    def extract_json(self, system, user):
        chunk = json.loads(user)["chunks"][0]
        return {"facts": [
            {"field": category, "text": f"The paper discusses {category} information for a research study.",
             "chunk_id": chunk["chunk_id"], "quote": chunk["text"][:90]}
            for category in ("limitation", "methodology", "dataset", "finding")
        ]}


class FakeComparisonLLM:
    model = "phase3-fake"

    def extract_json(self, system, user):
        facts = [x for x in json.loads(user)["facts"] if x["category"] == "limitation"]
        if len(facts) < 2:
            return {"insights": []}
        return {"insights": [{
            "kind": "shared_limitation", "statement": "Both selected sources identify a research limitation.",
            "fact_ids": [facts[0]["fact_id"], facts[1]["fact_id"]],
        }]}


def make_candidate(kind, ids):
    return {
        "kind": kind,
        "title": "Evaluate the reported limitations across academic systems",
        "rationale": "These paper-linked limitations motivate a focused experiment with comparable conditions.",
        "fact_ids": ids,
        "research_questions": [{"text": "How do different methods perform under comparable controlled conditions?"}],
        "importance": "high", "importance_reason": "The stated limitation affects the scope of conclusions.",
        "feasibility": "medium", "feasibility_reason": "A small study needs a controlled dataset and testing time.",
    }


class FakeGapLLM:
    model = "phase4-fake"

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.calls = []

    def extract_json(self, system, user):
        self.calls.append((system, user))
        if self.behavior == "api_error":
            raise LLMError("RATE_LIMITED", "Simulated rate limit")
        if self.behavior == "malformed_batch":
            return {"other": "not a gaps array"}
        if self.behavior == "empty":
            return {"gaps": []}
        facts = json.loads(user)["facts"]
        ids = lambda category: [f["fact_id"] for f in facts if f["category"] == category]
        limits, methods, datasets, findings = (ids(c) for c in ("limitation", "methodology", "dataset", "finding"))
        if not limits or not methods or not datasets or not findings:
            return {"gaps": []}
        gaps = [
            make_candidate("shared_limitation", limits[:2]),
            make_candidate("methodological_extension", [limits[0], methods[1]]),
            make_candidate("dataset_coverage", [limits[0], datasets[1]]),
            make_candidate("evaluation_opportunity", [limits[0], findings[1]]),
        ]
        if self.behavior == "unknown_id":
            gaps[0]["fact_ids"] = [limits[0], "fictional_fact_id"]
        if self.behavior == "same_paper":
            gaps[0]["fact_ids"] = [limits[0], methods[0]]
        if self.behavior == "shared_one_limitation":
            gaps[0]["fact_ids"] = [limits[0], methods[1]]
        if self.behavior == "wrong_category":
            gaps[0]["fact_ids"] = methods[:2]
        if self.behavior == "novelty":
            gaps[0]["title"] = "The first study of this completely unexplored research field"
        if self.behavior == "bad_question":
            gaps[0]["research_questions"] = [{"text": "Compare all the academic systems in the selected papers"}]
        if self.behavior == "duplicate":
            gaps.append(gaps[0].copy())
        if self.behavior == "invalid_item":
            gaps.insert(0, {"title": "missing other required fields"})
        if self.behavior == "oversized":
            gaps = [dict(gaps[i % 4], title=f"Candidate number {i} for a new controlled experiment")
                    for i in range(50)]
        return {"gaps": gaps}


@pytest.fixture
def input_comparison(three_papers):
    extracted = process_batch([PaperUpload(name, content) for name, content in three_papers])
    analyzed = analyze_batch(extracted, PaperAnalyzer(FakeEvidenceExtractor()))
    compared = compare_batch(analyzed, PaperComparator(FakeComparisonLLM()))
    assert compared.status == "complete"
    assert len(compared.comparison_table) == 3
    return compared


def test_phase4_complete_evidence_linked_gaps(input_comparison):
    llm = FakeGapLLM()
    result = discover_gaps(input_comparison, GapDiscoverer(llm))
    assert result.status == "complete"
    assert len(result.gaps) == 4
    assert result.paper_count == 3
    assert result.literature_verification == "not_performed"
    assert result.coverage.total_source_facts == 12
    assert result.coverage.facts_sent_to_llm == 12
    assert result.coverage.eligible_comparison_insights_used == 1
    assert result.coverage.truncated is False
    assert len(llm.calls) == 1
    assert "UNTRUSTED" in llm.calls[0][0]
    assert len({g.gap_id for g in result.gaps}) == 4
    for gap in result.gaps:
        assert gap.novelty_established is False
        assert gap.literature_verification == "not_performed"
        assert len({f.document_id for f in gap.cited_facts}) >= 2
        assert any(f.category == "limitation" for f in gap.cited_facts)
        assert len(gap.research_questions) in (1, 2)
        assert all(q.text.endswith("?") for q in gap.research_questions)
        for cited in gap.cited_facts:
            table_fact = next(
                f for row in input_comparison.comparison_table
                for attr in ("limitations", "methodologies", "datasets", "findings", "objectives", "contributions")
                for f in getattr(row, attr) if f.fact_id == cited.fact_id
            )
            assert cited == table_fact
    assert GapDiscoveryResult.model_validate_json(result.model_dump_json()) == result


def test_prompt_payload_and_insight_scope(input_comparison):
    llm = FakeGapLLM()
    GapDiscoverer(llm).discover(input_comparison)
    context = json.loads(llm.calls[0][1])
    assert context["scope"].startswith("Only these uploaded papers")
    assert len(context["papers"]) == 3
    assert len(context["facts"]) == 12
    assert len(context["comparison_insights"]) == 1
    assert set(context["facts"][0]) == {
        "fact_id", "document_id", "category", "statement", "source_page", "source_quote"
    }


@pytest.mark.parametrize("behavior", [
    "unknown_id", "same_paper", "shared_one_limitation", "wrong_category", "novelty", "bad_question",
])
def test_rejects_unsupported_first_gap_preserving_valid_others(input_comparison, behavior):
    result = GapDiscoverer(FakeGapLLM(behavior)).discover(input_comparison)
    assert len(result.gaps) == 3
    assert any("Rejected" in w for w in result.warnings)
    assert result.status == "complete"


def test_rejects_invalid_single_item_but_preserves_good_ones(input_comparison):
    result = GapDiscoverer(FakeGapLLM("invalid_item")).discover(input_comparison)
    assert len(result.gaps) == 4
    assert any("malformed" in w for w in result.warnings)


def test_deduplicates_same_references_and_kind(input_comparison):
    result = GapDiscoverer(FakeGapLLM("duplicate")).discover(input_comparison)
    assert len(result.gaps) == 4


def test_bounds_model_output(input_comparison):
    result = GapDiscoverer(FakeGapLLM("oversized")).discover(input_comparison)
    assert len(result.gaps) <= 5


@pytest.mark.parametrize("behavior", ["api_error", "malformed_batch"])
def test_llm_failure_does_not_fabricate_results(input_comparison, behavior):
    result = GapDiscoverer(FakeGapLLM(behavior)).discover(input_comparison)
    assert result.status == "partial"
    assert not result.gaps
    assert result.warnings


def test_empty_results_correctly_report_no_gap(input_comparison):
    result = GapDiscoverer(FakeGapLLM("empty")).discover(input_comparison)
    assert result.status == "no_grounded_gaps"
    assert not result.gaps


def test_no_llm_call_without_limitation_evidence(input_comparison):
    for row in input_comparison.comparison_table:
        row.limitations = []
    llm = FakeGapLLM()
    result = GapDiscoverer(llm).discover(input_comparison)
    assert result.status == "no_grounded_gaps"
    assert not llm.calls
    assert any("no model call" in x for x in result.warnings)


def test_no_llm_call_when_only_one_paper_has_facts(input_comparison):
    for row in input_comparison.comparison_table[1:]:
        for attr, _ in (("limitations", "limitation"), ("methodologies", "methodology"),
                        ("datasets", "dataset"), ("findings", "finding")):
            setattr(row, attr, [])
    llm = FakeGapLLM()
    result = GapDiscoverer(llm).discover(input_comparison)
    assert result.status == "no_grounded_gaps"
    assert not llm.calls


def test_partial_comparison_and_paper_analysis_status_are_disclosed(input_comparison):
    input_comparison.status = "partial"
    input_comparison.comparison_table[0].analysis_status = "partial"
    result = GapDiscoverer(FakeGapLLM()).discover(input_comparison)
    assert result.status == "partial"
    assert len(result.gaps) == 4
    assert any("partial" in w for w in result.warnings)


def test_context_truncation_with_explicit_warning(input_comparison):
    llm = FakeGapLLM()
    settings = GapSettings(max_context_characters=1600)
    result = GapDiscoverer(llm, settings).discover(input_comparison)
    assert result.coverage.truncated
    assert result.coverage.input_characters <= 1600
    assert any("Limited discovery context" in x for x in result.warnings)
    if llm.calls:
        assert len(llm.calls[0][1]) <= 1600


def test_source_table_rejects_duplicate_ids(input_comparison):
    input_comparison.comparison_table[1].limitations[0].fact_id = (
        input_comparison.comparison_table[0].limitations[0].fact_id
    )
    with pytest.raises(ValueError, match="Duplicate source fact"):
        validated_table_facts(input_comparison)


def test_source_table_rejects_bad_evidence_attribution(input_comparison):
    input_comparison.comparison_table[0].limitations[0].evidence.filename = "someone_else.pdf"
    with pytest.raises(ValueError, match="Source fact evidence"):
        GapDiscoverer(FakeGapLLM()).discover(input_comparison)


def test_source_table_rejects_wrong_category(input_comparison):
    input_comparison.comparison_table[0].limitations[0].category = "dataset"
    with pytest.raises(ValueError, match="Source fact evidence"):
        validated_table_facts(input_comparison)


def test_source_table_rejects_duplicate_papers(input_comparison):
    input_comparison.comparison_table[1].document_id = input_comparison.comparison_table[0].document_id
    with pytest.raises(ValueError, match="distinct"):
        GapDiscoverer(FakeGapLLM()).discover(input_comparison)


def test_source_table_rejects_count_mismatch(input_comparison):
    input_comparison.comparison_table.pop()
    with pytest.raises(ValueError, match="paper count"):
        GapDiscoverer(FakeGapLLM()).discover(input_comparison)


def test_node_supports_dictionary_state(input_comparison):
    node = GapDiscoveryNode(GapDiscoverer(FakeGapLLM()))
    patch = node({"paper_comparison": input_comparison.model_dump(mode="json")})
    assert "research_gaps" in patch
    assert len(patch["research_gaps"]["gaps"]) == 4


def test_config_validation_and_question_shape():
    with pytest.raises(ValueError):
        GapSettings(max_candidate_gaps=6)
    with pytest.raises(ValueError):
        GapSettings(max_context_characters=100)
    with pytest.raises(ValueError):
        GapSettings(max_facts_per_category_per_paper=11)
    with pytest.raises(ValueError):
        ResearchQuestion(text="This is not phrased as a proper research question")


def test_tampered_or_unselected_insights_not_passed_to_model(input_comparison):
    input_comparison.insights[0].cited_facts[0].statement = "TAMPERED FACT"
    llm = FakeGapLLM()
    GapDiscoverer(llm).discover(input_comparison)
    assert json.loads(llm.calls[0][1])["comparison_insights"] == []


def test_fact_selection_balance_across_papers(input_comparison):
    all_facts = validated_table_facts(input_comparison)
    selected, payload, count = select_gap_context(input_comparison, all_facts, GapSettings())
    assert len({fact.document_id for fact in selected.values()}) == 3
    assert len(payload) <= 24_000
    assert count == 1
