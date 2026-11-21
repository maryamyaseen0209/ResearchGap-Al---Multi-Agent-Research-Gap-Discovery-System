"""Generate print-ready Phase 7 example exports from fictional PDFs and fake AI.

This is an offline report/UI-export smoke test. It does NOT run LangGraph,
Groq, or OpenAlex, and does NOT validate any real research conclusion.

Run: python -m examples.demo_phase7_offline
Optional: python -m examples.demo_phase7_offline --output-dir /path/to/reports
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from examples.demo_phase4_offline import run as run_four_phases
from examples.demo_phase5_offline import OfflineOpenAlex, OfflineReviewModel
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.services.verification_service import verify_gaps
from researchgap.reports import generate_research_report


def run(output_dir: Path):
    processing, analyses, comparison, discovery = run_four_phases()
    verification = verify_gaps(discovery, GapVerifier(OfflineReviewModel(), OfflineOpenAlex()))
    report = generate_research_report(
        batch=processing,
        analyses=analyses,
        comparison=comparison,
        gaps=discovery,
        verification=verification,
        title="ResearchGap AI — Fictional Example Literature Report",
        author="ResearchGap AI demo (fictional data)",
        generated_at=datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc),
    )
    assert len(discovery.gaps) == len(verification.verifications) == 3
    assert report.markdown and report.html and report.json_text
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, text in (("md", report.markdown), ("html", report.html),
                            ("json", report.json_text)):
        (output_dir / f"{report.filename_stem}.{extension}").write_text(text, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("examples/sample_reports"))
    args = parser.parse_args(argv)
    export = run(args.output_dir)
    print("OFFLINE FICTIONAL DATA ONLY — no real LLM, OpenAlex, or novelty verification")
    print(f"Created 3 portable reports under {args.output_dir.resolve()}")
    print(f"Markdown: {len(export.markdown):,} chars, HTML: {len(export.html):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
