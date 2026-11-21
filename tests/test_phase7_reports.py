"""Phase 7 report content, citations, HTML sanitization, portability and CLI."""

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from researchgap.reports import generate_research_report
from researchgap.report_cli import main as report_cli_main
from researchgap.core.workflow_schemas import WorkflowResult
from test_workflow_service import make_runner, ScientificFakeLLM, EmptySearch
from langgraph_test_double import install
from researchgap.services.batch_processor import PaperUpload, process_batch


@pytest.fixture
def patch_graph(monkeypatch):
    install(monkeypatch)


@pytest.fixture
def processed_batch():
    location = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    return process_batch([PaperUpload(path.name, path.read_bytes())
                          for path in sorted(location.glob("*.pdf"))])


class HTMLAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.extend(x for k, x in attrs if k == "href")


@pytest.fixture
def report_data(patch_graph, processed_batch):
    result = make_runner(ScientificFakeLLM(), EmptySearch()).run(processed_batch)
    assert result.status == "complete"
    return processed_batch, result


def test_no_content_raises():
    with pytest.raises(ValueError, match="No research results"):
        generate_research_report()


def test_workflow_generates_three_consistent_offline_exports(report_data):
    batch, result = report_data
    outputs = generate_research_report(
        batch=batch, workflow=result, author="Research Team",
        generated_at=datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc),
    )
    parsed = json.loads(outputs.json_text)
    assert parsed["report_metadata"]["generated_at_utc"] == "2026-11-12 16:00 UTC"
    assert parsed["report_metadata"]["novelty_established"] is False
    assert parsed["report_metadata"]["human_review_required"] is True
    assert parsed["workflow"]["run_id"] == result.run_id
    assert len(parsed["processing"]["papers"]) == 3
    assert len(parsed["paper_analyses"]["analyses"]) == 3
    assert len(parsed["research_gaps"]["gaps"]) == 3
    assert "Cross-paper comparison" in outputs.markdown
    assert "Candidate research gaps" in outputs.markdown
    assert "External literature verification" in outputs.markdown
    assert "Agent execution trace" in outputs.markdown
    assert "Research Team" in outputs.markdown
    assert "Source:" in outputs.markdown
    assert "page 1" in outputs.markdown
    assert "print" in outputs.html
    assert "@media" in outputs.html
    assert "<html" in outputs.html
    html = HTMLAudit()
    html.feed(outputs.html)
    assert "main" in html.tags and "table" in html.tags
    assert "script" not in html.tags
    assert len(outputs.markdown) > 6000
    assert len(outputs.html) > 7000


def test_all_gap_evidence_quotes_resolve_in_original_pdf(report_data):
    batch, workflow = report_data
    pdfs = {p.document_id: p for p in batch.papers}
    report = generate_research_report(batch=batch, workflow=workflow)
    for gap in workflow.research_gaps.gaps:
        assert gap.title in report.markdown
        for fact in gap.cited_facts:
            page = pdfs[fact.document_id].pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
            assert " ".join(fact.evidence.quote.split()) in report.markdown
            assert any(source["evidence"]["quote"] == fact.evidence.quote
                       for g in json.loads(report.json_text)["research_gaps"]["gaps"]
                       for source in g["cited_facts"])


def test_html_and_markdown_escape_malicious_input(report_data):
    _, workflow = report_data
    poisoned = workflow.model_copy(deep=True)
    poisoned.events[0].message = '<script>alert("pwn")</script> [click](javascript:alert(1))'
    poisoned.paper_analyses.analyses[0].title = '<img src=x onerror="alert(1)">'
    report = generate_research_report(workflow=poisoned,
                                      title='"/><script>boom</script>')
    assert '<script>boom</script>' not in report.html
    assert '<img src=x' not in report.html
    assert "&lt;img" in report.html
    assert "&lt;script&gt;" in report.html
    assert "\\[click\\]" in report.markdown
    parsed = HTMLAudit()
    parsed.feed(report.html)
    assert "script" not in parsed.tags
    assert "img" not in parsed.tags
    assert parsed.links == []


def test_workflow_takes_precedence_over_inconsistent_manual_results(report_data):
    batch, result = report_data
    stale = result.research_gaps.model_copy(deep=True)
    stale.gaps[0].title = "THIS IS FROM A STALE RUN"
    artifacts = generate_research_report(batch=batch, workflow=result, gaps=stale)
    assert "THIS IS FROM A STALE RUN" not in artifacts.markdown
    assert json.loads(artifacts.json_text)["research_gaps"]["gaps"][0]["title"] == result.research_gaps.gaps[0].title


def test_partial_report_without_verification_warns(report_data):
    _, result = report_data
    artifacts = generate_research_report(gaps=result.research_gaps)
    assert "Literature verification not performed" in artifacts.markdown
    assert "NOT established" in artifacts.markdown
    data = json.loads(artifacts.json_text)
    assert data["literature_verification"] is None
    assert data["report_metadata"]["human_review_required"] is True


def test_generated_at_must_be_timezone_aware(report_data):
    _, result = report_data
    with pytest.raises(ValueError, match="timezone"):
        generate_research_report(workflow=result, generated_at=datetime(2026, 11, 12))


def test_report_cli_writes_three_outputs_without_api_keys(report_data, tmp_path, monkeypatch):
    batch, result = report_data
    src = tmp_path / "workflow.json"
    src.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    batch_file = tmp_path / "batch.json"
    batch_file.write_text(batch.model_dump_json(), encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    folder = tmp_path / "output"
    assert report_cli_main([str(src), "--batch", str(batch_file), "--output-dir", str(folder),
                            "--title", "Test Research Report"]) == 0
    md = folder / "researchgap_research_report.md"
    html = folder / "researchgap_research_report.html"
    structured = folder / "researchgap_research_report.json"
    assert all(p.is_file() for p in (md, html, structured))
    assert "Test Research Report" in md.read_text(encoding="utf-8")
    assert "<html" in html.read_text(encoding="utf-8")
    assert json.loads(structured.read_text(encoding="utf-8"))["report_metadata"]["novelty_established"] is False


def test_invalid_workflow_json_fails_with_helpful_error(tmp_path):
    src = tmp_path / "bad.json"
    src.write_text('{"not":"a-workflow"}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        report_cli_main([str(src), "--output-dir", str(tmp_path / "output")])
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


def test_report_serializes_partial_stopped_workflows_without_crashing(report_data):
    _, completed = report_data
    broken = WorkflowResult(
        run_id="partial-run", status="stopped", blocked_reason="Not enough valid papers",
        paper_analyses=completed.paper_analyses,
        events=completed.events[:2],
    )
    output = generate_research_report(workflow=broken)
    assert "Not enough valid papers" in output.markdown
    assert "Literature verification not performed" in output.markdown
    assert json.loads(output.json_text)["workflow"]["status"] == "stopped"


def test_scholarly_links_are_canonical_and_clickable_only_for_valid_work_urls():
    from examples.demo_phase5_offline import run as offline_phase5
    verification = offline_phase5()
    report = generate_research_report(
        gaps=verification.gap_discovery, verification=verification,
    )
    parser = HTMLAudit()
    parser.feed(report.html)
    assert parser.links
    assert all(link == "https://openalex.org/W10000000" for link in parser.links)
    assert "rel=\"noopener noreferrer\"" in report.html
    assert "[FICTIONAL study of limitations in academic systems]" in report.markdown
