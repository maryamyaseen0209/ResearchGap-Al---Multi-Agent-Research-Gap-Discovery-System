"""OFFLINE SIMULATION of Phases 1–3 on included PDFs (NOT a real AI evaluation).

Runs extraction, fact-to-quote linking, paper comparison, and cross-paper
citation checks without an API key or network request.
"""

import json
from pathlib import Path

from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.comparison_service import compare_batch


class OfflineSimulation:
    model = "offline-SIMULATION-not-an-LLM"

    def extract_json(self, system_message: str, user_message: str) -> dict:
        source = json.loads(user_message)
        if "chunks" in source:
            chunk = source["chunks"][0]
            return {"facts": [{
                "field": "methodology",
                "text": "Offline placeholder: a methodological passage exists in this source.",
                "chunk_id": chunk["chunk_id"],
                "quote": chunk["text"][:90],
            }]}
        first_two = [f["fact_id"] for f in source["facts"] if f["category"] == "methodology"][:2]
        return {"insights": [{
            "kind": "method_similarity",
            "statement": "Simulation only: example comparison, not a scientific conclusion.",
            "fact_ids": first_two,
        }] if len(first_two) == 2 else []}


def main() -> None:
    directory = Path(__file__).resolve().parent / "sample_papers"
    uploads = [PaperUpload(f.name, f.read_bytes()) for f in sorted(directory.glob("*.pdf"))]
    extracted = process_batch(uploads)
    simulator = OfflineSimulation()
    analyzed = analyze_batch(extracted, PaperAnalyzer(simulator))
    comparison = compare_batch(analyzed, PaperComparator(simulator))
    assert extracted.successful_count == 3
    assert analyzed.successful_count == 3
    assert comparison.paper_count == 3 and len(comparison.insights) == 1
    for insight in comparison.insights:
        assert len({fact.document_id for fact in insight.cited_facts}) >= 2
        for fact in insight.cited_facts:
            paper = next(p for p in extracted.papers if p.document_id == fact.document_id)
            page = paper.pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    print("OFFLINE SIMULATION — fabricated agent statements, NOT verified scientific findings")
    print(f"Phase 1: {extracted.successful_count} PDFs, {extracted.total_chunks} chunks")
    print(f"Phase 2: {analyzed.successful_count} analyses, {len(analyzed.failures)} failures")
    print(f"Phase 3: {comparison.paper_count} papers, {len(comparison.insights)} cited comparisons, ")
    print(f"         {comparison.coverage.facts_sent_to_llm} facts in comparison context")
    print("All simulated comparison citations resolve to original PDF page offsets.")


if __name__ == "__main__":
    main()
