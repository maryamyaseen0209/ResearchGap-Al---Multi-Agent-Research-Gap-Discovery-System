"""Offline export of a saved Phase 6 workflow JSON into Phase 7 reports.

No API keys, model calls, or external network requests are used.
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.core.schemas import BatchResult
from researchgap.reports import generate_research_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a source-grounded ResearchGap AI report")
    parser.add_argument("input", type=Path, help="Saved Phase 6 WorkflowResult JSON")
    parser.add_argument("--batch", type=Path, help="Optional Phase 1 PDF extraction JSON for paper inventory")
    parser.add_argument("--output-dir", type=Path, default=Path("researchgap_reports"))
    parser.add_argument("--title", default="ResearchGap AI — Evidence-Grounded Literature Review")
    parser.add_argument("--author", default="", help="Optional researcher / group name")
    args = parser.parse_args(argv)
    try:
        workflow = WorkflowResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        batch = (BatchResult.model_validate_json(args.batch.read_text(encoding="utf-8"))
                 if args.batch else None)
        exports = generate_research_report(batch=batch, workflow=workflow,
                                           title=args.title, author=args.author)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension, content in (
            ("md", exports.markdown), ("html", exports.html), ("json", exports.json_text)
        ):
            file_path = args.output_dir / f"{exports.filename_stem}.{extension}"
            file_path.write_text(content, encoding="utf-8")
            print(f"Created {file_path}")
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(f"Report generation failed: {type(exc).__name__}: {exc}")
    print("Reminder: this report cannot establish research novelty; review all sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
