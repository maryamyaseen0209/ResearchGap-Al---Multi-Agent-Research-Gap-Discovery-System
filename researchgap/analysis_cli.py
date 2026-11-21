"""Phase 2 CLI: analyze Phase 1 exported JSON with the Groq model."""

import argparse
from pathlib import Path
import os

from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.core.analysis_config import AnalysisSettings, configured_api_key
from researchgap.core.schemas import BatchResult
from researchgap.integrations.groq_client import GroqClient
from researchgap.services.analysis_service import analyze_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze research papers from Phase 1 JSON.")
    parser.add_argument("input", type=Path, help="Path to researchgap_extracted_papers.json")
    parser.add_argument("--output", type=Path, default=Path("researchgap_paper_analyses.json"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--max-chunks", type=int, default=24)
    args = parser.parse_args(argv)

    key = configured_api_key()
    if not key:
        parser.error("GROQ_API_KEY is missing. Put it in your .env file or environment.")
    try:
        batch = BatchResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        settings = AnalysisSettings(max_chunks_per_paper=args.max_chunks)
        analyzer = PaperAnalyzer(GroqClient(api_key=key, model=args.model), settings)
        result = analyze_batch(batch, analyzer)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Analyzed {result.successful_count}/{result.requested_count} papers; "
        f"{len(result.failures)} paper failures. Exported: {args.output}"
    )
    for analysis in result.analyses:
        print(f" - {analysis.filename}: {analysis.status}, {analysis.verified_fact_count} verified quotes")
    for failure in result.failures:
        print(f" - {failure.filename}: {failure.code} — {failure.message}")
    return 0 if not result.failures and all(a.status == "complete" for a in result.analyses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
