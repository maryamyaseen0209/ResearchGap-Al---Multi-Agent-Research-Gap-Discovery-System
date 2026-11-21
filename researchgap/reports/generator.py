"""Phase 7: deterministic, offline Markdown/HTML/JSON research reports.

The report is assembled solely from validated, existing phase outputs. It does
not ask an LLM to invent new findings or interpret absent evidence. HTML and
Markdown are escaped to avoid rendering untrusted paper text as active markup.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as html_escape
import json
import re
from typing import Any

from researchgap.core.schemas import BatchResult
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.core.comparison_schemas import ComparisonResult, CitedPaperFact
from researchgap.core.gap_schemas import GapDiscoveryResult
from researchgap.core.verification_schemas import LiteratureVerificationResult
from researchgap.core.workflow_schemas import WorkflowResult


@dataclass(frozen=True)
class ReportArtifacts:
    title: str
    markdown: str
    html: str
    json_text: str
    filename_stem: str = "researchgap_research_report"


@dataclass(frozen=True)
class Block:
    kind: str
    value: Any


def _clean(value: object) -> str:
    """One line, without invisible control characters or risky visual spacing."""
    text = str(value if value is not None else "")
    return " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())


def _md(value: object) -> str:
    text = _clean(value)
    for char in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#", "!", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def _safe_title(title: str) -> str:
    return _clean(title)[:120] or "ResearchGap AI — Evidence-Grounded Literature Review"


def _safe_author(author: str) -> str:
    return _clean(author)[:100]


def _append_fact(blocks: list[Block], fact: CitedPaperFact | Any, prefix: str = "") -> None:
    ev = fact.evidence
    blocks.append(Block("bullet", f"{prefix}{fact.statement}"))
    blocks.append(Block("quote", (
        f"Source: {ev.filename} · page {ev.page_number} · "
        f"chunk {ev.chunk_id} · characters {ev.start_char}–{ev.end_char} · "
        f"Exact excerpt: “{ev.quote}”"
    )))


def _render_markdown(blocks: list[Block]) -> str:
    out = []
    for block in blocks:
        v = block.value
        if block.kind == "h1":
            out.append(f"# {_md(v)}")
        elif block.kind == "h2":
            out.append(f"\n## {_md(v)}")
        elif block.kind == "h3":
            out.append(f"\n### {_md(v)}")
        elif block.kind == "text":
            out.append(_md(v))
        elif block.kind == "bullet":
            out.append(f"- {_md(v)}")
        elif block.kind == "quote":
            out.append(f"> {_md(v)}")
        elif block.kind == "study":
            name, year, url = v
            if re.fullmatch(r"https://openalex\.org/W[0-9]+", url):
                out.append(f"- Study: [{_md(name)}]({url}) ({_md(year)})")
            else:
                out.append(f"- Study: {_md(name)} ({_md(year)}) — {_md(url)}")
        elif block.kind == "table":
            headers, rows = v
            out.append("| " + " | ".join(_md(x) for x in headers) + " |")
            out.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                out.append("| " + " | ".join(_md(cell) for cell in row) + " |")
        else:
            raise ValueError(f"Unknown report block: {block.kind}")
    return "\n".join(out).strip() + "\n"


_HTML_STYLE = """
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif;color:#17233b}
*{box-sizing:border-box}body{background:#f3f6fc;margin:0;line-height:1.65}
.report{max-width:1040px;margin:44px auto;padding:64px 76px;background:white;border:1px solid #e5e9f3;border-radius:16px;box-shadow:0 18px 60px #18224415}
.kicker{font-size:12px;color:#6c55d7;letter-spacing:2.5px;text-transform:uppercase;font-weight:800}
h1{font-size:39px;line-height:1.16;color:#182444;letter-spacing:-1.5px;margin:22px 0 28px}
h2{font-size:23px;line-height:1.3;border-top:1px solid #e8ecf5;padding-top:32px;margin-top:37px;color:#1b2a50}
h3{font-size:17px;margin:24px 0 10px;color:#4242a4}a{color:#5e48bb;text-decoration:none;font-weight:650}a:hover{text-decoration:underline}p{margin:12px 0}li{padding:3px 0;overflow-wrap:anywhere}
blockquote{border-left:3px solid #9d85e9;background:#f7f5ff;margin:10px 0 18px;padding:12px 16px;border-radius:0 8px 8px 0;color:#4d5271;overflow-wrap:anywhere;font-size:13px}
.table-wrap{overflow:auto;border:1px solid #e9edf6;border-radius:10px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#f5f4ff;color:#414080;text-align:left}th,td{padding:11px 13px;border-bottom:1px solid #e7ecf4;vertical-align:top;overflow-wrap:anywhere}tr:last-child td{border-bottom:0}
.notice{background:#fffaee;border:1px solid #fae5aa;border-radius:10px;padding:14px 18px;color:#725b27;font-weight:600}
@media(max-width:750px){.report{margin:0;padding:25px 18px;border-radius:0}h1{font-size:29px}}
@media print{body{background:white}.report{box-shadow:none;border:0;margin:0;max-width:none;padding:0}h2,h3{break-after:avoid}blockquote,tr{break-inside:avoid}}
"""


def _render_html(blocks: list[Block], title: str) -> str:
    body = ['<main class="report">', '<div class="kicker">ResearchGap AI · Research Intelligence Report</div>']
    in_list = False
    for block in blocks:
        v = block.value
        if in_list and block.kind not in {"bullet", "study"}:
            body.append("</ul>")
            in_list = False
        if block.kind in {"h1", "h2", "h3"}:
            body.append(f"<{block.kind}>{html_escape(_clean(v), quote=True)}</{block.kind}>")
        elif block.kind == "text":
            safe = html_escape(_clean(v), quote=True)
            css = ' class="notice"' if "not establish" in _clean(v).lower() else ""
            body.append(f"<p{css}>{safe}</p>")
        elif block.kind == "bullet":
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{html_escape(_clean(v), quote=True)}</li>")
        elif block.kind == "study":
            if not in_list:
                body.append("<ul>")
                in_list = True
            name, year, url = v
            if re.fullmatch(r"https://openalex\.org/W[0-9]+", url):
                safe_url = html_escape(url, quote=True)
                body.append(f'<li>Study: <a href="{safe_url}" rel="noopener noreferrer">{html_escape(_clean(name),quote=True)}</a> ({html_escape(_clean(year),quote=True)})</li>')
            else:
                body.append(f"<li>Study: {html_escape(_clean(name),quote=True)} ({html_escape(_clean(year),quote=True)})</li>")
        elif block.kind == "quote":
            body.append(f"<blockquote>{html_escape(_clean(v), quote=True)}</blockquote>")
        elif block.kind == "table":
            headers, rows = v
            table = ['<div class="table-wrap"><table><thead><tr>']
            table.extend(f"<th>{html_escape(_clean(x), quote=True)}</th>" for x in headers)
            table.append("</tr></thead><tbody>")
            for row in rows:
                table.append("<tr>")
                table.extend(f"<td>{html_escape(_clean(x), quote=True)}</td>" for x in row)
                table.append("</tr>")
            table.append("</tbody></table></div>")
            body.extend(table)
        else:
            raise ValueError(f"Unknown report block: {block.kind}")
    if in_list:
        body.append("</ul>")
    body.append("</main>")
    safe_title = html_escape(_clean(title), quote=True)
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{safe_title}</title><style>{_HTML_STYLE}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")


def generate_research_report(
    *,
    batch: BatchResult | None = None,
    analyses: AnalysisBatchResult | None = None,
    comparison: ComparisonResult | None = None,
    gaps: GapDiscoveryResult | None = None,
    verification: LiteratureVerificationResult | None = None,
    workflow: WorkflowResult | None = None,
    title: str = "ResearchGap AI — Evidence-Grounded Literature Review",
    author: str = "",
    generated_at: datetime | None = None,
) -> ReportArtifacts:
    """Build a complete, evidence-traceable report without external services.

    When a workflow is supplied its stages take precedence to avoid combining
    stale manual-stage results with the most recent one-click run.
    """
    if workflow is not None:
        analyses, comparison, gaps, verification = (
            workflow.paper_analyses, workflow.paper_comparison,
            workflow.research_gaps, workflow.literature_verification,
        )
    if not any((batch, analyses, comparison, gaps, verification, workflow)):
        raise ValueError("No research results available to create a report")
    moment = generated_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    when = moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    name = _safe_title(title)
    researcher = _safe_author(author)
    blocks: list[Block] = [Block("h1", name)]
    blocks.append(Block("text", f"Generated: {when}" + (f" · Prepared by: {researcher}" if researcher else "")))
    blocks.append(Block("text", "Scientific novelty is NOT established by this report. Findings are limited to the uploaded papers and the recorded literature search. Human review is required."))

    docs = len(batch.papers) if batch else (comparison.paper_count if comparison else (analyses.successful_count if analyses else (gaps.paper_count if gaps else 0)))
    ngaps = len(gaps.gaps) if gaps else 0
    nchecks = len(verification.verifications) if verification else 0
    blocks.append(Block("h2", "Executive overview"))
    blocks.append(Block("table", (["Measure", "Result"], [
        ["Processed source PDFs", docs],
        ["Analyzed papers", analyses.successful_count if analyses else "Not available"],
        ["Cross-paper insights", len(comparison.insights) if comparison else "Not available"],
        ["Candidate research gaps", ngaps if gaps else "Not generated"],
        ["Literature verification assessments", nchecks if verification else "Not performed"],
        ["Workflow status", workflow.status if workflow else "Manual stage results"],
    ])))
    blocks.append(Block("text", "Method: PDF text extraction → source-quoted AI analysis → cited comparison → candidate gap discovery → limited OpenAlex literature verification. Missing data does not prove absence of research."))

    if batch:
        blocks.append(Block("h2", "Uploaded paper inventory"))
        blocks.append(Block("table", (["Document", "Pages", "Text pages", "Chunks", "Extraction warnings"], [
            [p.filename, p.page_count, p.text_page_count, len(p.chunks), "; ".join(p.warnings) or "None"]
            for p in batch.papers
        ])))
        for f in batch.failures:
            blocks.append(Block("bullet", f"Upload failure: {f.filename} — {f.code}: {f.message}"))

    if analyses:
        blocks.append(Block("h2", "Paper analyses and PDF evidence"))
        for a in analyses.analyses:
            blocks.append(Block("h3", f"{a.title} ({a.filename})"))
            blocks.append(Block("text", f"Status: {a.status} · Verified facts: {a.verified_fact_count} · Analyzed chunks: {a.coverage.analyzed_chunks}/{a.coverage.total_chunks} · Truncated: {a.coverage.truncated}"))
            categories = [
                ("research_objectives", "Objectives"), ("methodologies", "Methodologies"),
                ("datasets", "Datasets"), ("findings", "Findings"),
                ("limitations", "Limitations"), ("contributions", "Contributions"),
            ]
            for attr, label in categories:
                facts = getattr(a, attr)
                if facts:
                    blocks.append(Block("text", label))
                    for fact in facts:
                        _append_fact(blocks, fact)
            for warning in a.warnings:
                blocks.append(Block("bullet", f"Analysis warning: {warning}"))
        for failure in analyses.failures:
            blocks.append(Block("bullet", f"Analysis failure: {failure.filename} — {failure.code}: {failure.message}"))

    if comparison:
        blocks.append(Block("h2", "Cross-paper comparison"))
        rows = []
        for paper in comparison.comparison_table:
            rows.append([paper.title, paper.analysis_status] + [
                "; ".join(f.statement for f in getattr(paper, attr)) or "Not extracted"
                for attr in ("objectives", "methodologies", "datasets", "findings", "limitations", "contributions")
            ])
        blocks.append(Block("table", (["Paper", "Status", "Objectives", "Methodology", "Datasets", "Findings", "Limitations", "Contributions"], rows)))
        blocks.append(Block("text", f"Comparison status: {comparison.status}; source facts available to model: {comparison.coverage.facts_sent_to_llm}/{comparison.coverage.total_source_facts}."))
        for insight in comparison.insights:
            blocks.append(Block("h3", insight.kind.replace("_", " ").title()))
            blocks.append(Block("text", insight.statement))
            for fact in insight.cited_facts:
                _append_fact(blocks, fact, prefix=f"{fact.fact_id}: ")
        for warning in comparison.warnings:
            blocks.append(Block("bullet", f"Comparison warning: {warning}"))

    if gaps:
        blocks.append(Block("h2", "Candidate research gaps and questions"))
        blocks.append(Block("text", f"Discovery status: {gaps.status}. These are potential areas to investigate, not proven novel contributions."))
        for gap in gaps.gaps:
            blocks.append(Block("h3", f"{gap.gap_id} — {gap.title}"))
            blocks.append(Block("text", f"Category: {gap.kind.replace('_', ' ')} · Importance (heuristic): {gap.importance} · Feasibility (heuristic): {gap.feasibility}"))
            blocks.append(Block("text", f"Rationale: {gap.rationale}"))
            blocks.append(Block("text", f"Importance reasoning: {gap.importance_reason}. Feasibility reasoning: {gap.feasibility_reason}."))
            for q in gap.research_questions:
                blocks.append(Block("bullet", f"Research question: {q.text}"))
            for fact in gap.cited_facts:
                _append_fact(blocks, fact, prefix=f"{fact.fact_id}: ")
        for warning in gaps.warnings:
            blocks.append(Block("bullet", f"Gap discovery warning: {warning}"))

    if verification:
        blocks.append(Block("h2", "External literature verification (OpenAlex)"))
        blocks.append(Block("text", f"Verification run status: {verification.status}. A lack of direct matches does not establish novelty."))
        for v in verification.verifications:
            blocks.append(Block("h3", f"{v.gap_id}: {v.title}"))
            blocks.append(Block("text", f"Verdict: {v.status.replace('_', ' ')} · Searches: {v.searches_succeeded}/{v.searches_attempted} completed · Retrieved: {v.works_retrieved} · Reviewed: {v.works_reviewed}"))
            for query in v.search_queries:
                blocks.append(Block("bullet", f"Search query: {query}"))
            assessments = {a.openalex_id: a for a in v.assessments}
            for work in v.works:
                blocks.append(Block("study", (work.title, work.publication_year or "year unknown", work.url)))
                if work.openalex_id in assessments:
                    review = assessments[work.openalex_id]
                    blocks.append(Block("text", f"Assessment: {review.relevance}. {review.reasoning}"))
                    if review.evidence_quote:
                        blocks.append(Block("quote", f"Abstract/title excerpt: “{review.evidence_quote}”"))
            for warning in v.warnings:
                blocks.append(Block("bullet", f"Search warning: {warning}"))
        for warning in verification.warnings:
            blocks.append(Block("bullet", f"Verification warning: {warning}"))
    else:
        blocks.append(Block("h2", "Literature verification not performed"))
        blocks.append(Block("text", "No OpenAlex verification findings are included. Candidate research gaps have not been externally checked."))

    if workflow:
        blocks.append(Block("h2", "Agent execution trace"))
        blocks.append(Block("text", f"Workflow run ID: {workflow.run_id} · Status: {workflow.status} · Revisions attempted: {workflow.revision_count} · Revision applied: {workflow.revision_applied}"))
        if workflow.blocked_reason:
            blocks.append(Block("text", f"Blocked: {workflow.blocked_reason}"))
        for event in workflow.events:
            blocks.append(Block("bullet", f"{event.stage.upper()} [{event.status}]: {event.message}"))
        if workflow.initial_literature_verification and workflow.revision_attempted:
            blocks.append(Block("text", "An initial verification run triggered revision; the original decision is preserved in the attached JSON export."))

    blocks.append(Block("h2", "Limitations and responsible use"))
    for note in (
        "Exact quotations establish source provenance only, not scientific correctness.",
        "PDF extraction may omit figures, formulas, scanned text, or content outside chunk limits.",
        "Search results may be incomplete and retrieved abstracts may not capture full-paper conclusions.",
        "Candidate gaps require systematic literature review and expert assessment before claiming novelty.",
        "Automated importance and feasibility ratings are preliminary heuristics, not validated scores.",
    ):
        blocks.append(Block("bullet", note))

    payload = {
        "report_metadata": {"title": name, "author": researcher, "generated_at_utc": when,
                            "novelty_established": False, "human_review_required": True},
        "processing": batch.model_dump(mode="json") if batch else None,
        "paper_analyses": analyses.model_dump(mode="json") if analyses else None,
        "paper_comparison": comparison.model_dump(mode="json") if comparison else None,
        "research_gaps": gaps.model_dump(mode="json") if gaps else None,
        "literature_verification": verification.model_dump(mode="json") if verification else None,
        "workflow": workflow.model_dump(mode="json") if workflow else None,
    }
    return ReportArtifacts(
        title=name,
        markdown=_render_markdown(blocks),
        html=_render_html(blocks, name),
        json_text=json.dumps(payload, indent=2, ensure_ascii=False),
    )
