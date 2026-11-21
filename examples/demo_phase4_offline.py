"""End-to-end phases 1–4 using FICTIONAL PDFs and deterministic fake models.

No real LLM is called, no external literature is searched, and no scientific
conclusions or evidence of novelty are established by this simulation.
"""

import json
from pathlib import Path

from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.comparison_service import compare_batch
from researchgap.services.gap_service import discover_gaps


class OfflineFourPhaseModel:
    model = "OFFLINE-FICTIONAL-SIMULATION-NOT-AI"

    def extract_json(self, system_message: str, user_message: str) -> dict:
        data = json.loads(user_message)
        if "chunks" in data:  # Phase 2
            chunk = data["chunks"][0]
            text = chunk["text"]
            facts = []
            for token, field in (
                ("Limitation:", "limitation"), ("Methodology:", "methodology"),
                ("Dataset:", "dataset"), ("Finding:", "finding"),
            ):
                index = text.find(token)
                if index >= 0 and len(text) - index >= 12:
                    facts.append({
                        "field": field,
                        "text": f"FICTIONAL demonstration: a source passage contains {token}",
                        "chunk_id": chunk["chunk_id"],
                        "quote": text[index:min(index + 90, len(text))],
                    })
            return {"facts": facts}

        if "candidate_gap" in data and "works" in data:
            abstract_text = (
                "This fictional paper evaluates limitations in academic systems "
                "using a controlled follow-up experiment with multiple data sources."
            )
            return {"assessments": [
                {"openalex_id": w["openalex_id"], "relevance": "potentially_addresses",
                 "reasoning": "Offline demonstration: fictional abstract overlaps the simulated question.",
                 "evidence_quote": w.get("abstract") or abstract_text}
                for w in data.get("works", [])
            ]}

        facts_list = data.get("facts", [])
        fact_ids = lambda category: [
            f["fact_id"] for f in facts_list if f.get("category") == category
        ]
        if "comparison_insights" not in data:  # Phase 3
            limits = fact_ids("limitation")
            return {"insights": [{
                "kind": "shared_limitation",
                "statement": "FICTIONAL demonstration: two papers contain stated limitations.",
                "fact_ids": limits[:2],
            }] if len(limits) >= 2 else []}

        # Phase 4: these are fabricated candidate texts, but source quote IDs
        # are checked by the real Phase 4 validator.
        limits = fact_ids("limitation")
        methods = fact_ids("methodology")
        datasets = fact_ids("dataset")
        if len(limits) < 2 or len(methods) < 2 or len(datasets) < 2:
            return {"gaps": []}

        def candidate(kind, ids):
            return {
                "kind": kind,
                "title": "Fictional evaluation opportunity for follow-up research",
                "rationale": "Simulation only: source-quoted limitations suggest a topic to investigate further.",
                "fact_ids": ids,
                "research_questions": [{"text": "How do the approaches perform in a controlled follow-up experiment?"}],
                "importance": "medium",
                "importance_reason": "A larger evaluation might help clarify the noted limitations.",
                "feasibility": "medium",
                "feasibility_reason": "The study would require time, data, and external review.",
            }

        return {"gaps": [
            candidate("shared_limitation", limits[:2]),
            candidate("methodological_extension", [limits[0], methods[1]]),
            candidate("dataset_coverage", [limits[0], datasets[1]]),
        ]}


def run() -> tuple:
    source = Path(__file__).resolve().parent / "sample_papers"
    uploads = [PaperUpload(f.name, f.read_bytes()) for f in sorted(source.glob("*.pdf"))]
    model = OfflineFourPhaseModel()
    extracted = process_batch(uploads)
    analyzed = analyze_batch(extracted, PaperAnalyzer(model))
    compared = compare_batch(analyzed, PaperComparator(model))
    gaps = discover_gaps(compared, GapDiscoverer(model))

    assert extracted.successful_count == analyzed.successful_count == 3
    assert compared.paper_count == gaps.paper_count == 3
    assert len(compared.insights) == 1
    assert len(gaps.gaps) == 3 and gaps.status == "complete"
    for gap in gaps.gaps:
        assert gap.novelty_established is False
        assert gap.literature_verification == "not_performed"
        assert len({f.document_id for f in gap.cited_facts}) >= 2
        for fact in gap.cited_facts:
            paper = next(p for p in extracted.papers if p.document_id == fact.document_id)
            page = paper.pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    return extracted, analyzed, compared, gaps


def main() -> None:
    extracted, analyzed, compared, gaps = run()
    print("OFFLINE SIMULATION: All papers, LLM outputs, and proposed gaps are FICTIONAL")
    print(f"Phase 1: {extracted.successful_count} PDFs, {extracted.total_chunks} text chunks")
    print(f"Phase 2: {analyzed.successful_count} research paper analyses")
    print(f"Phase 3: {len(compared.insights)} cross-paper comparison")
    print(f"Phase 4: {len(gaps.gaps)} candidate gaps; status={gaps.status}")
    print("All Phase 4 citations exactly resolve to extracted PDF page positions")
    print("No scientific accuracy or global research novelty has been verified")


if __name__ == "__main__":
    main()
