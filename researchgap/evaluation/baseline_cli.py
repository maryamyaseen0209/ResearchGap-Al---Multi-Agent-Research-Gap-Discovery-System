"""One-off real-Groq baseline from a saved Phase 2 JSON (no extra paper upload)."""

import argparse
import os
from pathlib import Path

from researchgap.core.analysis_config import configured_api_key
from researchgap.core.analysis_schemas import AnalysisBatchResult
from researchgap.evaluation.baseline import SinglePassBaseline
from researchgap.evaluation.telemetry import MeteredLLM
from researchgap.integrations.groq_client import GroqClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Phase 8 single-pass research-gap baseline")
    parser.add_argument("--analyses", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("baseline_results.json"))
    args = parser.parse_args(argv)
    key = configured_api_key()
    if not key:
        parser.error("GROQ_API_KEY is required (see .env.example)")
    try:
        source = AnalysisBatchResult.model_validate_json(args.analyses.read_text(encoding="utf-8"))
        client = MeteredLLM(GroqClient(key, model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")))
        results = SinglePassBaseline(client).run(source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(results.model_dump_json(indent=2), encoding="utf-8")
        (args.output.parent / "baseline_telemetry.json").write_text(
            __import__("json").dumps(client.measurements(), indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Single-pass baseline: {len(results.gaps)} gaps, status={results.status}; saved {args.output}")
    return 0 if results.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
