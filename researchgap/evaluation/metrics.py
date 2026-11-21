"""Deterministic citation checks and human-review metrics, without fake accuracy.

Only actual page offsets and literal citations are automatically verifiable.
Whether a stated research gap is logically supported or genuinely novel
requires the annotation file and human judgment. Absent labels give NULL rates.
"""

from __future__ import annotations

import csv
from io import StringIO
from pydantic import BaseModel, Field

from researchgap.core.schemas import BatchResult
from researchgap.core.comparison_schemas import CitedPaperFact
from researchgap.core.workflow_schemas import WorkflowResult
from .baseline import BaselineResult


class SystemAudit(BaseModel):
    name: str
    gap_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    valid_page_citations: int = Field(ge=0)
    page_citation_integrity_rate: float | None = None
    gaps_citing_two_papers: int = Field(ge=0)
    cross_paper_citation_rate: float | None = None
    gaps_with_explicit_limitation: int = Field(ge=0)
    gaps_with_invalid_reference: int = Field(ge=0)
    gaps_with_novelty_claim_flag: int = Field(ge=0)
    human_reviewed_gaps: int = Field(ge=0)
    human_supported_gaps: int = Field(ge=0)
    human_support_rate: float | None = None
    human_unsupported_rate: float | None = None
    warnings: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    experiment: str = "Shared Phase 2 facts: one-pass baseline vs specialized agents with literature verification"
    baseline: SystemAudit
    multi_agent: SystemAudit
    human_labels_total: int
    interpretation: list[str]
    scientific_novelty_established: bool = False


def _page_match(fact: CitedPaperFact, batch: BatchResult) -> bool:
    paper = next((p for p in batch.papers if p.document_id == fact.document_id), None)
    ev = fact.evidence
    if not paper or paper.filename != fact.filename or ev.filename != paper.filename or ev.document_id != paper.document_id:
        return False
    if ev.page_number < 1 or ev.page_number > len(paper.pages):
        return False
    page = paper.pages[ev.page_number - 1]
    if not (0 <= ev.start_char < ev.end_char <= len(page.text)):
        return False
    return page.text[ev.start_char:ev.end_char] == ev.quote


def _novelty_flag(text: str) -> bool:
    from researchgap.agents.gap_discoverer import UNSUPPORTED_NOVELTY
    return bool(UNSUPPORTED_NOVELTY.search(text))


def _audit(
    name: str,
    items: list[tuple[str, str, list[CitedPaperFact], list[str]]],
    batch: BatchResult,
    labels: dict[tuple[str, str], bool],
) -> SystemAudit:
    citations = sum(len(facts) + len(invalid) for _, _, facts, invalid in items)
    valid = sum(_page_match(f, batch) for _, _, facts, _ in items for f in facts)
    cross_paper = sum(len({f.document_id for f in facts}) >= 2 for _, _, facts, _ in items)
    with_limit = sum(any(f.category == "limitation" for f in facts) for _, _, facts, _ in items)
    flawed = sum(bool(invalid) or any(not _page_match(f, batch) for f in facts)
                 for _, _, facts, invalid in items)
    flags = sum(_novelty_flag(title) for _, title, _, _ in items)
    reviews = [labels[(name, gid)] for gid, _, _, _ in items if (name, gid) in labels]
    n = len(items)
    audit = SystemAudit(name=name, gap_count=n, citation_count=citations,
        valid_page_citations=valid,
        page_citation_integrity_rate=(valid / citations if citations else None),
        gaps_citing_two_papers=cross_paper, cross_paper_citation_rate=(cross_paper / n if n else None),
        gaps_with_explicit_limitation=with_limit, gaps_with_invalid_reference=flawed,
        gaps_with_novelty_claim_flag=flags, human_reviewed_gaps=len(reviews),
        human_supported_gaps=sum(reviews),
        human_support_rate=(sum(reviews) / len(reviews) if reviews else None),
        human_unsupported_rate=(1 - sum(reviews) / len(reviews) if reviews else None))
    if not reviews:
        audit.warnings.append("No human judgments available: semantic support/accuracy cannot be scored.")
    if not citations:
        audit.warnings.append("No citations available: citation integrity is undefined.")
    return audit


def audit_workflow(batch: BatchResult, workflow: WorkflowResult, *, labels=None) -> SystemAudit:
    labels = labels or {}
    gaps = workflow.research_gaps.gaps if workflow.research_gaps else []
    return _audit("multi_agent", [(gap.gap_id, gap.title, gap.cited_facts, []) for gap in gaps], batch, labels)


def compare_systems(batch: BatchResult, baseline: BaselineResult, workflow: WorkflowResult, *, labels=None) -> EvaluationReport:
    labels = labels or {}
    source = [(g.gap_id, g.title, g.verified_citations, g.unresolved_fact_ids) for g in baseline.gaps]
    a = _audit("baseline", source, batch, labels)
    b = audit_workflow(batch, workflow, labels=labels)
    return EvaluationReport(baseline=a, multi_agent=b, human_labels_total=sum(
        1 for system in ("baseline", "multi_agent")
        for gid, _, _, _ in (source if system == "baseline" else [
            (g.gap_id, g.title, g.cited_facts, []) for g in (workflow.research_gaps.gaps if workflow.research_gaps else [])])
        if (system, gid) in labels), interpretation=[
            "Page quotation integrity is a mechanical check, NOT correctness of scientific interpretation.",
            "No direct OpenAlex match does not establish scientific novelty.",
            "Semantic support rates are only reported for explicitly human-labeled candidates.",
            "Baseline and multi-agent systems share Phase 2 extracted source facts; this is a downstream architecture comparison.",
            "Differences in model calls, tool access, and external literature searches must be disclosed separately."
        ])


def create_review_template(baseline: BaselineResult, workflow: WorkflowResult) -> str:
    """Generate CSV awaiting real human annotations; never pre-populate labels."""
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["system", "gap_id", "title", "supported_by_evidence", "reviewer_notes"])
    for gap in baseline.gaps:
        writer.writerow(["baseline", gap.gap_id, gap.title, "", ""])
    for gap in (workflow.research_gaps.gaps if workflow.research_gaps else []):
        writer.writerow(["multi_agent", gap.gap_id, gap.title, "", ""])
    return out.getvalue()


def read_review_labels(text: str) -> dict[tuple[str, str], bool]:
    labels = {}
    reader = csv.DictReader(StringIO(text))
    expected = {"system", "gap_id", "supported_by_evidence"}
    if not expected.issubset(reader.fieldnames or []):
        raise ValueError("Review CSV requires system, gap_id, supported_by_evidence columns")
    for row in reader:
        value = (row["supported_by_evidence"] or "").strip().lower()
        if not value:
            continue
        if value not in {"yes", "no"}:
            raise ValueError("Review labels must be 'yes', 'no', or empty")
        system, gid = row["system"], row["gap_id"]
        if system not in ("baseline", "multi_agent") or not gid:
            raise ValueError("Unknown system or blank gap_id in review CSV")
        key = (system, gid)
        if key in labels:
            raise ValueError("Duplicate annotated gap_id")
        labels[key] = value == "yes"
    return labels
