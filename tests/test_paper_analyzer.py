"""The LLM is simulated; only facts with exact source quotes are accepted."""
import json

import pytest

from researchgap.agents.paper_analyzer import PaperAnalyzer, extract_title, select_chunks, group_chunks
from researchgap.core.analysis_config import AnalysisSettings
from researchgap.core.schemas import BatchResult, ProcessedPaper
from researchgap.integrations.groq_client import LLMError
from researchgap.services.analysis_service import PaperAnalyzerNode, analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.tools.pdf_processor import process_pdf


class FakeLLM:
    model = "deterministic-test-LLM"

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.calls = []

    def extract_json(self, system, user):
        self.calls.append((system, user))
        source = json.loads(user)
        chunk = source["chunks"][0]
        quote = chunk["text"][:min(90, len(chunk["text"]))]
        if self.behavior == "invalid_quote":
            quote = "This quotation does not exist in the original PDF text."
        if self.behavior == "wrong_chunk":
            chunk["chunk_id"] = "paper-does-not-contain-this-chunk"
        if self.behavior == "api_error":
            raise LLMError("RATE_LIMITED", "Test-only: retry limit exceeded")
        if self.behavior == "invalid_schema":
            return {"facts": [{"field": "unknown_field", "text": "A claim", "quote": quote, "chunk_id": chunk["chunk_id"]}]}
        return {"facts": [{
            "field": "finding", "text": "The study reports a measurable improvement in performance.",
            "chunk_id": chunk["chunk_id"], "quote": quote,
        }]}


def test_analyzer_produces_source_verified_offsets(pdf_factory):
    data = pdf_factory(["This paper investigates multilingual education and reports new results. " * 2])
    paper = process_pdf("paper.pdf", data)
    llm = FakeLLM()
    analysis = PaperAnalyzer(llm).analyze(paper)
    assert analysis.status == "complete"
    assert analysis.verified_fact_count == 1
    fact = analysis.findings[0]
    assert fact.evidence.filename == "paper.pdf"
    assert fact.evidence.document_id == paper.document_id
    page = paper.pages[fact.evidence.page_number - 1]
    assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    assert fact.evidence.chunk_id in {c.chunk_id for c in paper.chunks}
    assert llm.calls and "UNTRUSTED DATA" in llm.calls[0][0]


@pytest.mark.parametrize("behavior", ["invalid_quote", "wrong_chunk"])
def test_fabricated_evidence_is_rejected(pdf_factory, behavior):
    paper = process_pdf("paper.pdf", pdf_factory(["Paper discusses genuine findings with measurable scientific results."]))
    analysis = PaperAnalyzer(FakeLLM(behavior)).analyze(paper)
    assert analysis.status == "no_verified_facts"
    assert not analysis.findings
    assert any("Rejected" in warning for warning in analysis.warnings)


def test_malformed_output_handled_with_partial_result(pdf_factory):
    paper = process_pdf("paper.pdf", pdf_factory(["Paper investigates a reproducible research evaluation."]))
    analysis = PaperAnalyzer(FakeLLM("invalid_schema")).analyze(paper)
    assert analysis.status == "partial"
    assert analysis.coverage.batches_succeeded == 0
    assert "invalid extraction fields" in analysis.warnings[0]


def test_api_error_keeps_status_and_failure_warning(pdf_factory):
    paper = process_pdf("paper.pdf", pdf_factory(["Paper investigates a reproducible research evaluation."]))
    analysis = PaperAnalyzer(FakeLLM("api_error")).analyze(paper)
    assert analysis.status == "partial"
    assert analysis.coverage.batches_attempted == 1
    assert analysis.coverage.batches_succeeded == 0
    assert "RATE_LIMITED" in analysis.warnings[0]


def test_batch_multiple_papers_and_langgraph_node(three_papers):
    batch = process_batch([PaperUpload(name, content) for name, content in three_papers])
    analyzer = PaperAnalyzer(FakeLLM())
    result = analyze_batch(batch, analyzer)
    assert result.successful_count == 3
    assert result.failures == []
    assert all(a.verified_fact_count == 1 for a in result.analyses)
    state_patch = PaperAnalyzerNode(analyzer)({"batch": batch.model_dump(mode="json")})
    assert len(state_patch["paper_analyses"]["analyses"]) == 3
    assert BatchResult.model_validate(batch).ready_for_analysis


def test_selection_discloses_skipped_chunks(pdf_factory):
    text = "Methodology: We conduct repeatable machine-learning experiments. " * 12
    paper = process_pdf("many.pdf", pdf_factory([text, text, text]))
    assert len(paper.chunks) >= 3
    settings = AnalysisSettings(max_chunks_per_paper=1, max_characters_per_paper=2500,
                                max_characters_per_request=2000)
    analysis = PaperAnalyzer(FakeLLM(), settings).analyze(paper)
    assert analysis.coverage.truncated
    assert analysis.status == "partial"
    assert analysis.coverage.analyzed_chunks == 1
    assert any("Partial coverage" in w for w in analysis.warnings)


def test_chunk_grouping_and_limits(pdf_factory):
    paper = process_pdf("paper.pdf", pdf_factory([
        "Relevant first-page scientific methodology. " * 5,
        "Relevant second-page scientific conclusions. " * 5,
    ]))
    groups = group_chunks(paper.chunks, 250)
    assert len(groups) >= 2
    assert sum(len(g) for g in groups) == len(paper.chunks)
    assert all(sum(len(c.text) for c in g) <= 250 for g in groups)
    with pytest.raises(ValueError, match="exceeds"):
        group_chunks(paper.chunks, 10)


def test_title_fallbacks(pdf_factory):
    paper = process_pdf("paper.pdf", pdf_factory(["A very clear paper headline appears right at the start."]))
    paper.title = None
    assert extract_title(paper)[1] == "first_page"
    paper.pages[0].text = "Short"
    assert extract_title(paper) == ("paper", "filename")


def test_settings_reject_invalid_limits():
    with pytest.raises(ValueError):
        AnalysisSettings(max_characters_per_request=15_000, max_characters_per_paper=14_000)


def test_truncated_output_no_unsupported_claim_of_full_coverage(pdf_factory):
    data = pdf_factory(["Abstract and methodology describe research work. " * 14,
                        "Results and limitations describe evaluation concerns. " * 14])
    paper = process_pdf("sample.pdf", data)
    settings = AnalysisSettings(max_chunks_per_paper=1, max_characters_per_paper=6000,
                                max_characters_per_request=3000)
    result = PaperAnalyzer(FakeLLM(), settings).analyze(paper)
    assert result.coverage.analyzed_chunks <= 1
    assert result.coverage.total_chunks >= result.coverage.analyzed_chunks
    assert result.coverage.truncated == (result.coverage.total_chunks > 1)
