"""Comprehensive regression tests for API reliability, persistent batch caching,
correct coverage accounting, semantic evidence validation, and title extraction.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
import pytest

from researchgap.agents.batch_cache import AnalysisBatchCache, compute_batch_key, compute_chunk_content_hash
from researchgap.agents.paper_analyzer import (
    PaperAnalyzer,
    extract_title,
    validate_semantic_support,
    deduplicate_facts,
    compute_unique_character_coverage,
)
from researchgap.core.analysis_config import AnalysisSettings
from researchgap.core.analysis_schemas import (
    AnalysisBatchResult,
    AnalysisFailure,
    CandidateExtraction,
    EvidenceReference,
    GroundedFact,
    PaperAnalysis,
    PaperCoverage,
)
from researchgap.core.schemas import BatchResult, PageText, ProcessedPaper, TextChunk
from researchgap.integrations.failover_client import FailoverLLMClient, KeyTarget
from researchgap.integrations.groq_client import JSONLanguageModel, LLMError
from researchgap.tools.pdf_processor import strip_authors_from_title, is_author_or_metadata_text, format_title_casing


class MockLLM:
    """Mock LLM that can simulate success, rate limits, quota exhaustion, or sequential responses."""

    def __init__(self, model: str = "mock-model", responses: list[Any] | None = None):
        self.model = model
        self.responses = list(responses or [])
        self.call_count = 0
        self.received_messages: list[tuple[str, str]] = []

    def extract_json(self, system_message: str, user_message: str) -> dict:
        self.call_count += 1
        self.received_messages.append((system_message, user_message))
        if not self.responses:
            return {"facts": []}
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def make_dummy_paper(doc_id: str, filename: str, pages_text: list[str]) -> ProcessedPaper:
    """Create a test ProcessedPaper with deterministic chunks."""
    pages = [PageText(page_number=i + 1, text=t, character_count=len(t)) for i, t in enumerate(pages_text)]
    total_chars = sum(len(t) for t in pages_text)
    chunks = []
    chunk_idx = 1
    for page in pages:
        # 2 chunks per page
        mid = len(page.text) // 2
        c1 = TextChunk(
            chunk_id=f"{doc_id}_c{chunk_idx}",
            document_id=doc_id,
            page_number=page.page_number,
            chunk_index=chunk_idx,
            start_char=0,
            end_char=mid,
            text=page.text[:mid],
        )
        chunk_idx += 1
        c2 = TextChunk(
            chunk_id=f"{doc_id}_c{chunk_idx}",
            document_id=doc_id,
            page_number=page.page_number,
            chunk_index=chunk_idx,
            start_char=mid,
            end_char=len(page.text),
            text=page.text[mid:],
        )
        chunk_idx += 1
        chunks.extend([c1, c2])

    return ProcessedPaper(
        document_id=doc_id,
        sha256="dummy_sha256_hash_" + doc_id,
        filename=filename,
        file_size_bytes=max(1024, total_chars),
        page_count=len(pages),
        text_page_count=len(pages),
        total_characters=total_chars,
        title=None,
        pages=pages,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# 1. API Rate Limits & Cooldowns
# ---------------------------------------------------------------------------

def test_temporary_rate_limit_triggers_cooldown_and_resumes():
    """Verify temporary 429 rate limit triggers provider cooldown and resets after timeout."""
    g1 = MockLLM("gemini-2.5-flash", [LLMError("RATE_LIMITED", "RPM limit reached", retry_after=0.2)])
    targets = [KeyTarget("gemini", 1, g1, "gem1")]
    client = FailoverLLMClient(targets=targets)

    # First call hits rate limit and sets cooldown
    with pytest.raises(LLMError) as exc_info:
        client.extract_json("sys", "user")
    assert exc_info.value.code == "RATE_LIMITED"
    assert "cooling down" in str(exc_info.value) or "rate limits" in str(exc_info.value)

    # Cooldown is active
    assert client.cooldown_until.get("gemini", 0) > time.time()

    # Reset transient limits clears cooldown for resuming
    client.reset_transient_limits()
    assert client.cooldown_until.get("gemini", 0) <= time.time()


def test_permanent_quota_exhaustion_marks_provider_exhausted_and_shifts():
    """Verify permanent quota exhaustion on Gemini shifts to Groq and skips Gemini on subsequent requests."""
    g1 = MockLLM("gemini-2.5-flash", [LLMError("QUOTA_EXHAUSTED", "Daily quota reached")])
    q1 = MockLLM("openai/gpt-oss-120b", [{"facts": []}])
    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("groq", 1, q1, "groq1"),
    ]
    client = FailoverLLMClient(targets=targets)

    # First request shifts from Gemini to Groq
    res = client.extract_json("sys", "user")
    assert res == {"facts": []}
    assert client.active_provider == "groq"
    assert client.is_provider_exhausted("gemini") is True

    # Second request immediately goes to Groq without calling exhausted Gemini
    q1.responses.append({"facts": []})
    res2 = client.extract_json("sys", "user")
    assert res2 == {"facts": []}
    assert g1.call_count == 1  # Gemini was never called a second time


# ---------------------------------------------------------------------------
# 2. Persistent Batch Caching & Resume
# ---------------------------------------------------------------------------

def test_batch_caching_and_resume_across_restarts(tmp_path):
    """Test partial analysis where 1 batch succeeds and 1 fails; on resume only remaining batch is called."""
    import hashlib
    cache = AnalysisBatchCache(cache_dir=tmp_path / "cache")
    p1 = "Page 1: CRAG retrieval evaluator achieves 54.9% accuracy on PopQA with LLaMA2-7B. " * 3
    p2 = "Page 2: Standard RAG baseline achieved only 44.3% accuracy on the same benchmark. " * 3
    paper = make_dummy_paper("doc1", "test.pdf", [p1, p2])

    batch1_facts = {
        "facts": [{
            "field": "finding",
            "text": "CRAG achieved 54.9% accuracy on PopQA with LLaMA2-7B.",
            "chunk_id": "doc1_c1",
            "quote": "CRAG retrieval evaluator achieves 54.9% accuracy on PopQA with LLaMA2-7B.",
        }]
    }

    settings = AnalysisSettings(max_characters_per_request=300)
    # Run 1: Batch 1 succeeds, Batch 2 fails with rate limit
    mock_llm_1 = MockLLM("gemini-2.5-flash", [batch1_facts, LLMError("RATE_LIMITED", "429 Rate Limit")])
    analyzer_1 = PaperAnalyzer(mock_llm_1, settings=settings, cache=cache, enable_cache=True)
    res_1 = analyzer_1.analyze(paper, resume=True)

    assert res_1.is_partial is True
    assert res_1.coverage.batches_succeeded == 1
    assert res_1.coverage.batches_failed == 1
    assert res_1.verified_fact_count == 1
    assert mock_llm_1.call_count == 2

    # Check cache status
    groups = [paper.chunks[:2], paper.chunks[2:]]
    dhash = hashlib.sha256("".join(p.text for p in paper.pages).encode("utf-8")).hexdigest()[:16]
    progress = cache.get_paper_progress(paper.document_id, dhash, groups)
    assert progress["cached_batches"] == 1
    assert progress["pending_batches"] == 1

    # Run 2: Resume analysis with fresh LLM; Batch 1 should be loaded from cache, Batch 2 executed
    batch2_facts = {
        "facts": [{
            "field": "finding",
            "text": "Standard RAG baseline achieved 44.3% accuracy on the same benchmark.",
            "chunk_id": "doc1_c3",
            "quote": "Standard RAG baseline achieved only 44.3% accuracy on the same benchmark.",
        }]
    }
    mock_llm_2 = MockLLM("gemini-2.5-flash", [batch2_facts])
    analyzer_2 = PaperAnalyzer(mock_llm_2, settings=settings, cache=cache, enable_cache=True)
    res_2 = analyzer_2.analyze(paper, resume=True)

    assert res_2.coverage.batches_cached == 1
    assert res_2.coverage.batches_succeeded == 2
    assert res_2.verified_fact_count == 2
    # Only 1 LLM request was made because Batch 1 was loaded from cache!
    assert mock_llm_2.call_count == 1


def test_cache_reuse_across_model_failover(tmp_path):
    """Ensure cache hits are preserved when failover shifts model (e.g. Gemini -> Groq)."""
    cache = AnalysisBatchCache(cache_dir=tmp_path / "cache")
    paper = make_dummy_paper("doc2", "crag.pdf", ["Page 1 text content for caching."])

    extraction = CandidateExtraction.model_validate({"facts": []})
    chunk_ids = [c.chunk_id for c in paper.chunks]
    chash = compute_chunk_content_hash(paper.chunks)
    key = compute_batch_key(paper.document_id, "hash1", chunk_ids, chash)

    # Save under Gemini
    cache.save_cached_batch(key, extraction, model="gemini-2.5-flash", provider="gemini")

    # Read back under Groq
    data = cache.get_cached_batch_data(key)
    assert data is not None
    assert data["model"] == "gemini-2.5-flash"
    assert data["provider"] == "gemini"


# ---------------------------------------------------------------------------
# 3. Correct Coverage & Batch Accounting
# ---------------------------------------------------------------------------

def test_correct_coverage_accounting_pending_vs_failed(tmp_path):
    """Replicate Self-RAG accounting: 1 batch failed, 0 succeeded; pending chunks must NOT be counted as failed."""
    cache = AnalysisBatchCache(cache_dir=tmp_path / "cache")
    paper = make_dummy_paper(
        "doc3", "self_rag.pdf",
        ["Text content on page " + str(i) + " of Self-RAG research paper. " * 5 for i in range(1, 7)]
    )

    settings = AnalysisSettings(max_characters_per_request=350)
    # Simulate immediate rate limit on first batch
    mock_llm = MockLLM("gemini-2.5-flash", [LLMError("RATE_LIMITED", "Rate limit exceeded")])
    analyzer = PaperAnalyzer(mock_llm, settings=settings, cache=cache, enable_cache=True)
    analysis = analyzer.analyze(paper, resume=True)

    cov = analysis.coverage
    assert cov.batches_attempted == 1
    assert cov.batches_succeeded == 0
    assert cov.batches_failed == 1
    assert cov.pending_batches > 0

    # Chunks accounting
    assert cov.successful_chunks == 0
    assert cov.failed_chunks == cov.attempted_chunks
    assert cov.pending_chunks == cov.selected_chunks - cov.attempted_chunks
    # Total selected must equal attempted + pending
    assert cov.selected_chunks == cov.attempted_chunks + cov.pending_chunks
    # Unique analyzed characters must be 0 since 0 batches succeeded
    assert cov.unique_analyzed_characters == 0


def test_zero_successful_batches_marks_analysis_failed_and_records_failure():
    """Verify a paper with zero successful batches is marked is_failed=True and recorded in failures."""
    paper = make_dummy_paper("doc4", "rag.pdf", ["Content page 1", "Content page 2"])
    mock_llm = MockLLM("gemini-2.5-flash", [LLMError("RATE_LIMITED", "429 Quota Exceeded")])
    analyzer = PaperAnalyzer(mock_llm, enable_cache=False)

    from researchgap.services.analysis_service import analyze_batch
    batch = BatchResult(requested_count=1, papers=[paper], failures=[])
    result = analyze_batch(batch, analyzer)

    assert result.successful_count == 0
    assert result.failed_count == 1
    assert len(result.failures) == 1
    assert result.failures[0].filename == "rag.pdf"
    assert result.analyses[0].is_failed is True


# ---------------------------------------------------------------------------
# 4. Scientific Evidence Validation (Two-Stage Verification)
# ---------------------------------------------------------------------------

def test_scientific_support_validation_catches_misleading_claims():
    """Verify missing numbers or vague fragments are not labeled as fully supported."""
    # Case 1: Statement claims specific number '54.9%' not in quotation
    status1, reason1 = validate_semantic_support(
        statement="CRAG achieved 54.9% accuracy on PopQA with LLaMA2-7B.",
        quote="Results on four datasets demonstrate that CRAG outperforms standard RAG.",
        category="finding",
    )
    assert status1 == "requires_human_review"
    assert "54.9" in reason1

    # Case 2: Very short or heading-only quotation
    status2, reason2 = validate_semantic_support(
        statement="CRAG significantly improves standard RAG performance.",
        quote="Results on four datasets",
        category="finding",
    )
    assert status2 == "partially_supported"
    assert "isolated heading" in reason2

    # Case 3: Fully grounded statement with matching numbers and predicate
    status3, reason3 = validate_semantic_support(
        statement="CRAG achieved 54.9% accuracy on PopQA with LLaMA2-7B, outperforming standard RAG at 44.3%.",
        quote="CRAG achieved 54.9% accuracy on PopQA with LLaMA2-7B, outperforming standard RAG which scored 44.3%.",
        category="finding",
    )
    assert status3 == "supported"
    assert "supports key concepts" in reason3


# ---------------------------------------------------------------------------
# 5. Multiline Title Extraction
# ---------------------------------------------------------------------------

def test_multiline_title_extraction_skips_arxiv_header_and_excludes_authors():
    """Verify title extractor skips arXiv/preprint header lines and stops before author names."""
    page_text = (
        "arXiv:2310.11511v1 [cs.CL] 17 Oct 2023\n"
        "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND\n"
        "CRITIQUE THROUGH SELF-REFLECTION\n"
        "Akari Asai†*, Zeqiu Wu†, Yizhong Wang†, Avirup Sil‡, Hannaneh Hajishirzi†§\n"
        "†University of Washington ‡IBM Research §Allen Institute for AI\n"
        "Abstract\n"
        "Despite their remarkable capabilities, large language models..."
    )
    paper = make_dummy_paper("doc5", "paper2_self_rag.pdf", [page_text])

    title, source = extract_title(paper)
    assert title == "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
    assert source == "first_page"
    assert "Akari Asai" not in title
    assert "arXiv" not in title


# ---------------------------------------------------------------------------
# 6. Category-Aware Deduplication
# ---------------------------------------------------------------------------

def test_category_aware_deduplication_merges_duplicate_objectives():
    """Verify semantically duplicate research objectives are merged while distinct benchmarks are kept."""
    ev1 = EvidenceReference(document_id="d1", filename="f1.pdf", chunk_id="c1", page_number=1, start_char=0, end_char=50, quote="We investigate improving retrieval in RAG.")
    ev2 = EvidenceReference(document_id="d1", filename="f1.pdf", chunk_id="c2", page_number=1, start_char=60, end_char=110, quote="We evaluate improving retrieval in RAG.")
    ev3 = EvidenceReference(document_id="d1", filename="f1.pdf", chunk_id="c3", page_number=2, start_char=0, end_char=60, quote="CRAG achieved 54.9% accuracy on PopQA.")
    ev4 = EvidenceReference(document_id="d1", filename="f1.pdf", chunk_id="c4", page_number=2, start_char=70, end_char=130, quote="CRAG achieved 62.1% accuracy on TriviaQA.")

    facts = [
        GroundedFact(category="research_objective", statement="Investigate improving retrieval in RAG models.", evidence=ev1),
        GroundedFact(category="research_objective", statement="Investigate improving retrieval quality in RAG models.", evidence=ev2),
        GroundedFact(category="finding", statement="CRAG achieved 54.9% accuracy on PopQA benchmark.", evidence=ev3),
        GroundedFact(category="finding", statement="CRAG achieved 62.1% accuracy on TriviaQA benchmark.", evidence=ev4),
    ]

    deduped, count = deduplicate_facts(facts)
    assert count == 1  # 1 duplicate objective merged
    assert len(deduped) == 3

    # Check that the two findings (PopQA vs TriviaQA) were preserved
    finding_statements = [f.statement for f in deduped if f.category == "finding"]
    assert len(finding_statements) == 2
    assert any("PopQA" in s for s in finding_statements)
    assert any("TriviaQA" in s for s in finding_statements)
