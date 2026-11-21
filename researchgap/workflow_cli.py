"""Run full LangGraph phases 2–6 from a saved Phase 1 JSON extraction."""

import argparse
import os
from time import perf_counter
import json
from researchgap.evaluation.telemetry import MeteredLLM, MeteredSearch
from pathlib import Path

from pydantic import ValidationError

from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.analysis_config import configured_api_key
from researchgap.core.schemas import BatchResult
from researchgap.integrations.groq_client import GroqClient
from researchgap.integrations.openalex_client import OpenAlexClient
from researchgap.services.workflow_service import ResearchWorkflow, WorkflowDependencyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unified ResearchGap AI LangGraph workflow")
    parser.add_argument("input", type=Path, help="Phase 1 extracted papers JSON")
    parser.add_argument("--output", type=Path, default=Path("researchgap_workflow_report.json"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--no-revision", action="store_true", help="Skip the optional challenged-gap revision")
    args = parser.parse_args(argv)

    key = configured_api_key()
    if not key:
        parser.error("Missing GROQ_API_KEY; set it in .env (do not commit secrets).")
    try:
        batch = BatchResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        model = MeteredLLM(GroqClient(api_key=key, model=args.model))
        search = MeteredSearch(OpenAlexClient(api_key=os.getenv("OPENALEX_API_KEY", "").strip() or None))
        runner = ResearchWorkflow(
            PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
            GapVerifier(model, search), enable_revision=not args.no_revision,
        )
        began = perf_counter()
        result = runner.run(batch)
        telemetry = {"workflow_wall_seconds": round(perf_counter()-began, 4),
                     "llm": model.measurements(), "literature_search": search.measurements()}
    except (OSError, ValueError, ValidationError, WorkflowDependencyError) as exc:
        parser.error(f"Cannot run workflow: {type(exc).__name__}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    telemetry_path = args.output.with_name(args.output.stem + "_telemetry.json")
    telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print(f"Workflow {result.status}; {len(result.events)} events; revision attempts={result.revision_count}.")
    print(f"Saved report to {args.output}")
    print("Human review required: scoped literature checking does not prove scientific novelty.")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
