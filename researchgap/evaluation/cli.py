"""Read completed runs and create transparent machine/hand-review evaluation."""

import argparse
import json
from pathlib import Path

from researchgap.core.schemas import BatchResult
from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.evaluation.baseline import BaselineResult
from researchgap.evaluation.metrics import (
    compare_systems, create_review_template, read_review_labels,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a baseline and a multi-agent workflow against source PDFs")
    parser.add_argument("--batch", type=Path, required=True, help="Phase 1 JSON")
    parser.add_argument("--baseline", type=Path, required=True, help="Single-pass baseline JSON")
    parser.add_argument("--workflow", type=Path, required=True, help="Phase 6 workflow JSON")
    parser.add_argument("--labels", type=Path, help="Optional human-review CSV (yes/no/blank)")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_output"))
    args = parser.parse_args(argv)
    try:
        batch = BatchResult.model_validate_json(args.batch.read_text(encoding="utf-8"))
        base = BaselineResult.model_validate_json(args.baseline.read_text(encoding="utf-8"))
        workflow = WorkflowResult.model_validate_json(args.workflow.read_text(encoding="utf-8"))
        labels = read_review_labels(args.labels.read_text(encoding="utf-8")) if args.labels else {}
        allowed = {("baseline", g.gap_id) for g in base.gaps} | {
            ("multi_agent", g.gap_id) for g in (workflow.research_gaps.gaps if workflow.research_gaps else [])
        }
        if not set(labels).issubset(allowed):
            raise ValueError("Review CSV contains gap IDs not present in supplied results")
        summary = compare_systems(batch, base, workflow, labels=labels)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_results.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (args.output_dir / "human_review_template.csv").write_text(create_review_template(base, workflow), encoding="utf-8")
    table = [
        "# ResearchGap AI — Evaluation", "",
        "**Study type:** Same Phase 2 evidence, downstream single-pass vs multi-agent architecture.",
        "", "| Metric | Single-pass baseline | Multi-agent |", "|---|---:|---:|",
    ]
    for label, key in (
        ("Candidate gaps", "gap_count"),
        ("Verified page citations", "valid_page_citations"),
        ("Citation integrity rate", "page_citation_integrity_rate"),
        ("Gaps with invalid citations", "gaps_with_invalid_reference"),
        ("Gaps citing two papers", "gaps_citing_two_papers"),
        ("Human-reviewed gaps", "human_reviewed_gaps"),
        ("Human-evaluated support rate", "human_support_rate"),
    ):
        def display(obj):
            value = getattr(obj, key)
            if value is None: return "Not measured"
            return f"{value:.1%}" if key.endswith("rate") else str(value)
        table.append(f"| {label} | {display(summary.baseline)} | {display(summary.multi_agent)} |")
    table += ["", "## Limitations", "", *[f"- {sentence}" for sentence in summary.interpretation],
              "", "**This report does not establish scientific novelty or benchmark superiority.**", ""]
    (args.output_dir / "evaluation_summary.md").write_text("\n".join(table), encoding="utf-8")
    print(f"Evaluation written to {args.output_dir}; labeled gaps: {summary.human_labels_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
