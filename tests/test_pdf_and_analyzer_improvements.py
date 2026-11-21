"""Regression tests for two-column PDF extraction, multiline titles, and evidence recovery."""
import json
import pymupdf
import pytest

from researchgap.agents.paper_analyzer import (
    PaperAnalyzer,
    extract_title,
    format_title_casing,
    locate_evidence_in_chunk,
    select_chunks,
)
from researchgap.core.analysis_config import AnalysisSettings
from researchgap.integrations.groq_client import JSONLanguageModel
from researchgap.tools.pdf_processor import (
    extract_page_text_layout_aware,
    extract_title_from_pdf,
    process_pdf,
)


class MockRecoveryLLM:
    """Mock LLM returning slightly formatted or whitespace-altered quotes."""
    model = "mock-recovery-llm"

    def __init__(self, quote_generator):
        self.quote_generator = quote_generator

    def extract_json(self, system: str, user: str) -> dict:
        source = json.loads(user)
        chunk = source["chunks"][0]
        quote, field, text = self.quote_generator(chunk)
        return {
            "facts": [
                {
                    "field": field,
                    "text": text,
                    "chunk_id": chunk["chunk_id"],
                    "quote": quote,
                }
            ]
        }


def test_two_column_reading_order_preservation():
    """Verify that text from Column 1 and Column 2 are not interleaved row-by-row."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    # Left column: rect (50, 100, 280, 700)
    c1_lines = "First column sentence one.\nFirst column sentence two.\nFirst column sentence three."
    page.insert_textbox(pymupdf.Rect(50, 100, 280, 700), c1_lines)
    # Right column: rect (320, 100, 550, 700)
    c2_lines = "Second column sentence one.\nSecond column sentence two.\nSecond column sentence three."
    page.insert_textbox(pymupdf.Rect(320, 100, 550, 700), c2_lines)

    extracted = extract_page_text_layout_aware(page)

    # Column 1 should appear in full BEFORE Column 2 begins
    c1_pos = extracted.find("First column sentence three.")
    c2_pos = extracted.find("Second column sentence one.")
    assert c1_pos != -1
    assert c2_pos != -1
    assert c1_pos < c2_pos, "Column 1 must precede Column 2 in reading order!"

    # Ensure sentences are not interleaved on the same lines
    lines = extracted.splitlines()
    for line in lines:
        assert not ("First column" in line and "Second column" in line)


def test_mixed_layout_with_spanning_title_and_footer():
    """Verify that a page with a spanning title, two columns, and a footer orders correctly."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    # Title (spanning): (50, 40, 550, 80)
    page.insert_textbox(pymupdf.Rect(50, 40, 550, 80), "Title: Complete Research Paper Name Across Columns")
    # Column 1: (50, 120, 280, 600)
    page.insert_textbox(pymupdf.Rect(50, 120, 280, 600), "Left column paragraph about methodology.")
    # Column 2: (320, 120, 550, 600)
    page.insert_textbox(pymupdf.Rect(320, 120, 550, 600), "Right column paragraph about experimental results.")
    # Footer: (50, 740, 550, 770)
    page.insert_textbox(pymupdf.Rect(50, 740, 550, 770), "Page 1 - Academic Conference Proceedings")

    extracted = extract_page_text_layout_aware(page)
    title_idx = extracted.find("Title: Complete Research Paper")
    left_idx = extracted.find("Left column paragraph")
    right_idx = extracted.find("Right column paragraph")
    footer_idx = extracted.find("Page 1 - Academic Conference")

    assert title_idx < left_idx < right_idx < footer_idx


def test_multiline_title_extraction_from_pdf():
    """Verify that paper titles spanning multiple lines are fully extracted."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    # Insert multiline title with large font (18pt)
    page.insert_textbox(
        pymupdf.Rect(50, 40, 550, 120),
        "Retrieval-Augmented Generation for\nKnowledge-Intensive NLP Tasks",
        fontsize=18,
    )
    # Insert author names with smaller font (11pt)
    page.insert_textbox(
        pymupdf.Rect(50, 130, 550, 160),
        "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni",
        fontsize=11,
    )
    # Insert abstract text (10pt)
    page.insert_textbox(
        pymupdf.Rect(50, 180, 550, 300),
        "Abstract: Large pre-trained language models have been shown to store vast factual knowledge.",
        fontsize=10,
    )

    extracted_title = extract_title_from_pdf(doc)
    assert extracted_title is not None
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" in extracted_title
    assert "Patrick Lewis" not in extracted_title


def test_all_caps_title_formatting():
    """Verify that ALL CAPS titles are formatted to Title Case while keeping acronyms."""
    raw = "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION"
    formatted = format_title_casing(raw)
    assert "Self-RAG" in formatted
    assert "Learning" in formatted
    assert "Self-Reflection" in formatted


def test_evidence_locator_recovers_newline_quotes():
    """Verify that quotes differing only by whitespace/newlines in the PDF are safely recovered."""
    chunk_text = (
        "We propose Retrieval-Augmented Generation (RAG)\n"
        "combining pre-trained parametric and non-parametric memory\n"
        "for knowledge-intensive NLP tasks."
    )
    # LLM emitted spaces instead of \n
    llm_quote = "We propose Retrieval-Augmented Generation (RAG) combining pre-trained parametric"
    located = locate_evidence_in_chunk(chunk_text, llm_quote)
    assert located is not None
    exact_quote, start, end = located
    # The recovered quote strictly matches chunk_text slice!
    assert chunk_text[start:end] == exact_quote
    assert exact_quote.startswith("We propose Retrieval-Augmented Generation (RAG)\ncombining")


def test_evidence_locator_recovers_unicode_quotes_and_dashes():
    """Verify that quotes with ASCII hyphens/quotes match Unicode em-dashes and smart quotes."""
    chunk_text = (
        "On PopQA — an open-domain benchmark — CRAG achieved 54.9% accuracy,\n"
        "outperforming standard RAG models."
    )
    llm_quote = "On PopQA - an open-domain benchmark - CRAG achieved 54.9% accuracy"
    located = locate_evidence_in_chunk(chunk_text, llm_quote)
    assert located is not None
    exact_quote, start, end = located
    assert chunk_text[start:end] == exact_quote
    assert "—" in exact_quote


def test_evidence_locator_strictly_rejects_hallucinations():
    """Verify that genuinely nonexistent or fabricated text is strictly rejected."""
    chunk_text = "The model was trained on 4 NVIDIA A100 GPUs using AdamW optimizer."
    hallucinated_quote = "The model achieved 99.9% state of the art accuracy on all tasks."
    assert locate_evidence_in_chunk(chunk_text, hallucinated_quote) is None


def test_analyzer_end_to_end_recovers_evidence_with_accurate_offsets(pdf_factory):
    """Verify that PaperAnalyzer accepts facts whose quote had a newline and computes exact offsets."""
    text = (
        "Corrective Retrieval Augmented Generation introduces\n"
        "a lightweight retrieval evaluator to assess document quality."
    )
    paper = process_pdf("paper.pdf", pdf_factory([text]))

    # LLM provides quote without the \n
    def quote_gen(chunk):
        return (
            "Corrective Retrieval Augmented Generation introduces a lightweight retrieval evaluator",
            "methodology",
            "CRAG introduces a retrieval evaluator to assess document quality.",
        )

    llm = MockRecoveryLLM(quote_gen)
    analysis = PaperAnalyzer(llm).analyze(paper)

    assert analysis.status == "complete"
    assert len(analysis.methodologies) == 1
    fact = analysis.methodologies[0]
    # Check that evidence quote was recovered with exact newline from source
    assert "\n" in fact.evidence.quote
    page = paper.pages[fact.evidence.page_number - 1]
    assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    assert any("Evidence validation summary" in w for w in analysis.warnings)


def test_section_aware_chunk_selection_excludes_references(pdf_factory):
    """Verify that chunks from the References section are excluded from analysis."""
    pages = [
        "1. Introduction: Retrieval systems often struggle with hallucination.",
        "2. Methodology: We propose an adaptive self-reflection mechanism.",
        "3. Experiments and Results: On PopQA, our model achieves 54.9 accuracy.",
        "References\n[1] Lewis et al. 2020. RAG.\n[2] Asai et al. 2023. Self-RAG.\n[3] Yan et al. 2024. CRAG.",
        "[4] Vaswani et al. 2017. Attention.\n[5] Devlin et al. 2019. BERT.\n[6] Brown et al. 2020. GPT-3.",
    ]
    paper = process_pdf("paper.pdf", pdf_factory(pages))
    settings = AnalysisSettings(max_chunks_per_paper=3)
    selected = select_chunks(paper, settings)

    # Chunks from the bibliography should not crowd out core sections
    selected_texts = [c.text for c in selected]
    assert any("Methodology" in t or "Introduction" in t for t in selected_texts)
