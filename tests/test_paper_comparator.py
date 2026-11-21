"""Phase 3 agent behavior with deterministic fake models, no internet required."""
import json

import pytest

from researchgap.agents.paper_comparator import (
    FACT_FIELDS, SYSTEM_PROMPT, PaperComparator, build_comparison_table,
    select_facts_for_llm,
)
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.comparison_config import ComparisonSettings
from researchgap.core.comparison_schemas import ComparisonResult
from researchgap.integrations.groq_client import LLMError
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.comparison_service import PaperComparisonNode, compare_batch


class SourceAwareExtractor:
    model = "phase2-fake"

    def extract_json(self, system, user):
        data = json.loads(user)
        chunk = data["chunks"][0]
        quote = chunk["text"][: min(45, len(chunk["text"]))]
        category = {
            "a.pdf": "methodology", "b.pdf": "methodology", "c.pdf": "methodology",
            "d.pdf": "methodology", "e.pdf": "methodology",
        }.get(data["filename"], "finding")
        return {"facts": [{"field": category, "text": f"Source details for {data['filename']} are presented.",
                           "chunk_id": chunk["chunk_id"], "quote": quote}]}


class ComparisonFakeLLM:
    model = "phase3-fake"

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.calls = []

    def extract_json(self, system, user):
        self.calls.append((system, user))
        if self.behavior == "llm_error":
            raise LLMError("RATE_LIMITED", "Simulated rate limit")
        if self.behavior == "invalid_schema":
            return {"insights": [{"kind": "does_not_exist", "statement": "invalid kind", "fact_ids": []}]}
        if self.behavior == "empty":
            return {"insights": []}
        payload = json.loads(user)
        sources = [x for x in payload["facts"] if x["category"] == "methodology"]
        if len(sources) < 2:
            return {"insights": []}
        ids = [sources[0]["fact_id"], sources[1]["fact_id"]]
        if self.behavior == "unknown_id":
            ids[1] = "invented_id"
        if self.behavior == "same_paper":
            ids[1] = ids[0]
        if self.behavior == "wrong_category":
            return {"insights": [{"kind": "shared_limitation", "statement": "Methods share the same limitation.", "fact_ids": ids}]}
        if self.behavior == "one_paper_twice":
            ids = [ids[0], ids[0], ids[0]]
        candidate = {"kind": "method_similarity",
                     "statement": "Both papers include specific reported methodological procedures.",
                     "fact_ids": ids}
        return {"insights": [candidate, candidate] if self.behavior == "duplicate" else [candidate]}


@pytest.fixture
def compared_analyses(three_papers):
    batch = process_batch([PaperUpload(name, content) for name, content in three_papers])
    return analyze_batch(batch, PaperAnalyzer(SourceAwareExtractor()))


def test_comparison_builds_auditable_table_and_grounded_cross_paper_insight(compared_analyses):
    llm = ComparisonFakeLLM()
    result = compare_batch(compared_analyses, PaperComparator(llm))
    assert result.status == "complete"
    assert result.paper_count == 3
    assert len(result.comparison_table) == 3
    assert len(result.insights) == 1
    assert result.coverage.total_source_facts == 3
    assert result.coverage.facts_sent_to_llm == 3
    assert len(llm.calls) == 1
    assert "UNTRUSTED" in llm.calls[0][0]
    insight = result.insights[0]
    assert len({citation.document_id for citation in insight.cited_facts}) == 2
    for source in insight.cited_facts:
        matching_paper = next(x for x in compared_analyses.analyses if x.document_id == source.document_id)
        original_fact = matching_paper.methodologies[0]
        assert source.evidence == original_fact.evidence
        assert source.evidence.filename == original_fact.evidence.filename
        assert source.statement == original_fact.statement
    assert ComparisonResult.model_validate_json(result.model_dump_json()) == result


def test_table_uses_only_phase2_evidence(compared_analyses):
    table, facts = build_comparison_table(compared_analyses.analyses)
    assert len(facts) == 3
    assert all(row.analysis_status == "complete" for row in table)
    assert all(not row.limitations for row in table)
    assert all(row.methodologies[0].evidence.quote for row in table)
    assert all(row.document_id == row.methodologies[0].evidence.document_id for row in table)
    assert len({fact.fact_id for fact in facts}) == 3


@pytest.mark.parametrize("behavior", ["unknown_id", "same_paper", "one_paper_twice", "wrong_category"])
def test_rejects_invalid_cross_paper_citations(compared_analyses, behavior):
    result = PaperComparator(ComparisonFakeLLM(behavior)).compare(compared_analyses)
    assert result.status == "no_supported_comparisons"
    assert result.insights == []
    assert any("Rejected" in warning for warning in result.warnings)
    assert len(result.comparison_table) == 3


def test_duplicate_insights_are_deduplicated(compared_analyses):
    result = PaperComparator(ComparisonFakeLLM("duplicate")).compare(compared_analyses)
    assert len(result.insights) == 1


@pytest.mark.parametrize("behavior", ["llm_error", "invalid_schema"])
def test_model_failure_retains_factual_comparison_table(compared_analyses, behavior):
    result = PaperComparator(ComparisonFakeLLM(behavior)).compare(compared_analyses)
    assert result.status == "partial"
    assert result.insights == []
    assert len(result.comparison_table) == 3
    assert result.warnings


def test_no_model_call_when_no_facts(compared_analyses):
    for analysis in compared_analyses.analyses:
        analysis.methodologies = []
        analysis.status = "no_verified_facts"
    llm = ComparisonFakeLLM()
    result = PaperComparator(llm).compare(compared_analyses)
    assert not llm.calls
    assert result.status == "no_supported_comparisons"
    assert result.coverage.total_source_facts == 0
    assert any("Insufficient" in warning for warning in result.warnings)


def test_partial_coverage_and_failed_preceding_paper_reported(compared_analyses):
    compared_analyses.analyses[0].status = "partial"
    from researchgap.core.analysis_schemas import AnalysisFailure
    compared_analyses.failures.append(
        AnalysisFailure(document_id="other-id", filename="bad.pdf", code="ERROR", message="unavailable")
    )
    result = PaperComparator(ComparisonFakeLLM()).compare(compared_analyses)
    assert result.status == "partial"
    assert len(result.insights) == 1
    assert any("failed paper" in warning for warning in result.warnings)
    assert any("missing or partial" in warning for warning in result.warnings)


def test_prompt_payload_contains_only_scoped_facts(compared_analyses):
    llm = ComparisonFakeLLM()
    PaperComparator(llm).compare(compared_analyses)
    payload = json.loads(llm.calls[0][1])
    assert len(payload["papers"]) == 3
    assert len(payload["facts"]) == 3
    assert all(set(item) == {"fact_id", "paper_id", "category", "statement", "source_quote", "source_page"}
               for item in payload["facts"])


def test_duplicate_paper_ids_are_rejected(compared_analyses):
    compared_analyses.analyses[1].document_id = compared_analyses.analyses[0].document_id
    with pytest.raises(ValueError, match="unique"):
        PaperComparator(ComparisonFakeLLM()).compare(compared_analyses)


def test_fewer_than_3_and_more_than_5_are_rejected(compared_analyses):
    with pytest.raises(ValueError, match="3–5"):
        PaperComparator(ComparisonFakeLLM()).compare(
            AnalysisBatchResult(requested_count=2, analyses=compared_analyses.analyses[:2])
        )
    six = [p.model_copy(deep=True) for p in compared_analyses.analyses] * 2
    with pytest.raises(ValueError, match="3–5"):
        PaperComparator(ComparisonFakeLLM()).compare(AnalysisBatchResult(requested_count=6, analyses=six))


def test_rejects_cross_document_or_wrong_category_attribution(compared_analyses):
    compared_analyses.analyses[0].methodologies[0].evidence.document_id = "other"
    with pytest.raises(ValueError, match="Inconsistent"):
        PaperComparator(ComparisonFakeLLM()).compare(compared_analyses)


def test_rejects_wrong_filename_attribution(compared_analyses):
    compared_analyses.analyses[0].methodologies[0].evidence.filename = "other.pdf"
    with pytest.raises(ValueError, match="Inconsistent"):
        PaperComparator(ComparisonFakeLLM()).compare(compared_analyses)


def test_schemas_validate_configuration_bounds():
    with pytest.raises(ValueError):
        ComparisonSettings(max_context_characters=100)
    with pytest.raises(ValueError):
        ComparisonSettings(max_facts_per_category_per_paper=11)
    with pytest.raises(ValueError):
        ComparisonSettings(max_insights=0)


def test_context_truncation_is_explicit_and_table_is_not_truncated(compared_analyses):
    for item in compared_analyses.analyses:
        for i in range(6):
            fact = item.methodologies[0].model_copy(deep=True)
            fact.statement = f"Repeated methodology description {i} " + ("long scientific detail " * 18)
            item.methodologies.append(fact)
    settings = ComparisonSettings(max_context_characters=1400, max_facts_per_category_per_paper=2)
    llm = ComparisonFakeLLM()
    result = PaperComparator(llm, settings).compare(compared_analyses)
    assert result.coverage.truncated
    assert result.coverage.facts_sent_to_llm < result.coverage.total_source_facts
    assert sum(len(row.methodologies) for row in result.comparison_table) == 21
    assert any("Limited LLM comparison context" in w for w in result.warnings)
    if llm.calls:
        assert len(llm.calls[0][1]) <= 1400


def test_fact_selection_balances_paper_coverage(compared_analyses):
    table, _ = build_comparison_table(compared_analyses.analyses)
    selection, payload = select_facts_for_llm(table, ComparisonSettings(max_context_characters=12000))
    assert len(selection) == 3
    assert len({f.document_id for f in selection.values()}) == 3
    assert len(payload) <= 12000


def test_no_llm_call_when_only_one_paper_has_facts(compared_analyses):
    for item in compared_analyses.analyses[1:]:
        item.methodologies = []
    llm = ComparisonFakeLLM()
    result = PaperComparator(llm).compare(compared_analyses)
    assert result.status == "no_supported_comparisons"
    assert not llm.calls


def test_langgraph_ready_node_supports_dict_input(compared_analyses):
    node = PaperComparisonNode(PaperComparator(ComparisonFakeLLM()))
    patch = node({"paper_analyses": compared_analyses.model_dump(mode="json")})
    assert "paper_comparison" in patch
    assert len(patch["paper_comparison"]["comparison_table"]) == 3
    assert len(patch["paper_comparison"]["insights"]) == 1


def test_all_fact_categories_are_supported_in_table(compared_analyses):
    fact = compared_analyses.analyses[0].methodologies[0]
    for source_attr, table_attr, category in FACT_FIELDS:
        copy = fact.model_copy(deep=True)
        copy.category = category
        setattr(compared_analyses.analyses[0], source_attr, [copy])
    table, facts = build_comparison_table(compared_analyses.analyses)
    for source_attr, table_attr, category in FACT_FIELDS:
        assert len(getattr(table[0], table_attr)) == 1
        assert getattr(table[0], table_attr)[0].category == category
    assert len(facts) == 8
