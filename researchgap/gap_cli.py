"""Run Phase 4 on a Phase 3 exported JSON file; needs GROQ_API_KEY."""

import argparse
import os
from pathlib import Path

from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.core.analysis_config import configured_api_key
from researchgap.core.comparison_schemas import ComparisonResult
from researchgap.integrations.groq_client import GroqClient
from researchgap.services.gap_service import discover_gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover evidence-linked candidate research gaps")
    parser.add_argument("input", type=Path, help="JSON output from researchgap.comparison_cli")
    parser.add_argument("--output", type=Path, default=Path("researchgap_candidate_gaps.json"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    args = parser.parse_args(argv)
    key = configured_api_key()
    if not key:
        parser.error("GROQ_API_KEY is missing. Set it in .env or the environment.")
    try:
        comparison = ComparisonResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        result = discover_gaps(comparison, GapDiscoverer(GroqClient(api_key=key, model=args.model)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Processed {result.paper_count} papers; {len(result.gaps)} evidence-linked candidate "
        f"gaps; status={result.status}; external verification=not performed; output={args.output}"
    )
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
