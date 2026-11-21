"""Run Phase 3 on a Phase 2 exported JSON file; requires GROQ_API_KEY."""

import argparse
import os
from pathlib import Path

from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.analysis_config import configured_api_key
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.integrations.groq_client import GroqClient
from researchgap.services.comparison_service import compare_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare 3–5 analyzed research papers")
    parser.add_argument("input", type=Path, help="Output JSON from researchgap.analysis_cli")
    parser.add_argument("--output", type=Path, default=Path("researchgap_paper_comparison.json"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    args = parser.parse_args(argv)

    key = configured_api_key()
    if not key:
        parser.error("GROQ_API_KEY is missing. Set it in .env or the environment.")
    try:
        batch = AnalysisBatchResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        result = compare_batch(batch, PaperComparator(GroqClient(api_key=key, model=args.model)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Compared {result.paper_count} papers; {len(result.insights)} grounded "
        f"insights; status={result.status}; output={args.output}"
    )
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
