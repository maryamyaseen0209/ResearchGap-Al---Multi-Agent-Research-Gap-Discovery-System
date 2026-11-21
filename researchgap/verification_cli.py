"""Verify a saved Phase 4 JSON report against OpenAlex and Groq."""

import argparse
import os
from pathlib import Path

from researchgap.agents.gap_verifier import GapVerifier
from researchgap.core.analysis_config import configured_api_key
from researchgap.core.gap_schemas import GapDiscoveryResult
from researchgap.integrations.groq_client import GroqClient
from researchgap.integrations.openalex_client import OpenAlexClient
from researchgap.services.verification_service import verify_gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check candidate gaps against external scholarly literature")
    parser.add_argument("input", type=Path, help="Phase 4 candidate gaps JSON")
    parser.add_argument("--output", type=Path, default=Path("researchgap_verification.json"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    args = parser.parse_args(argv)
    key = configured_api_key()
    if not key:
        parser.error("GROQ_API_KEY is missing. Set it in .env or the environment.")
    try:
        discovery = GapDiscoveryResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        verifier = GapVerifier(
            GroqClient(api_key=key, model=args.model),
            OpenAlexClient(api_key=os.getenv("OPENALEX_API_KEY", "") or None),
        )
        result = verify_gaps(discovery, verifier)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    counts = {kind: sum(g.status == kind for g in result.verifications) for kind in (
        "challenged", "no_direct_match_found", "inconclusive"
    )}
    print(f"OpenAlex verification {result.status}: {counts} -> {args.output}")
    print("WARNING: Search results and model judgements cannot prove global novelty.")
    return 0 if result.status in ("complete", "no_gaps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
