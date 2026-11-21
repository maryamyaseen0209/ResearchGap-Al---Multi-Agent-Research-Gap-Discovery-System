"""EXPLICIT MOCK DEMO: exercises both phases without contacting an AI model.

These synthetic outputs do not demonstrate LLM extraction quality.
Use the real Groq analysis command for actual model-driven results.
"""

import json
from pathlib import Path

from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch


class OfflineModelStub:
    model = "offline-test-stub-NOT-AI"

    def extract_json(self, system_message: str, user_message: str) -> dict:
        first = json.loads(user_message)["chunks"][0]
        return {"facts": [{
            "field": "research_objective",
            "text": "Demonstration only: extracted first passage of the source document.",
            "chunk_id": first["chunk_id"],
            "quote": first["text"][:min(110, len(first["text"]))],
        }]}


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    folder = project_dir / "examples" / "sample_papers"
    inputs = [PaperUpload(file.name, file.read_bytes()) for file in sorted(folder.glob("*.pdf"))]
    batch = process_batch(inputs)
    result = analyze_batch(batch, PaperAnalyzer(OfflineModelStub()))
    print("OFFLINE SIMULATION ONLY: no live LLM or scientific claims are validated.")
    print(f"Phase 1: {batch.successful_count} PDFs, {batch.total_chunks} chunks")
    print(f"Phase 2: {result.successful_count} analyses, {len(result.failures)} failures")
    for analysis in result.analyses:
        evidence = analysis.research_objectives[0].evidence
        assert analysis.verified_fact_count
        source = next(p for p in batch.papers if p.document_id == analysis.document_id)
        page = source.pages[evidence.page_number - 1]
        assert page.text[evidence.start_char:evidence.end_char] == evidence.quote
        print(f" - {analysis.filename}: {analysis.status}, quote verified on page {evidence.page_number}")


if __name__ == "__main__":
    main()
