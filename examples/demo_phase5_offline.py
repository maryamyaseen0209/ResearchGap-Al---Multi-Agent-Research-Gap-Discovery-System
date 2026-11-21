"""Offline five-phase integration demonstration with FICTIONAL PDF and work data.

No Groq or OpenAlex HTTP requests are made. Do not treat the output as a real
scientific conclusion, novelty search, or independently validated citation.
"""

import json

from examples.demo_phase4_offline import run as run_four_phases
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.core.verification_schemas import LiteratureWork
from researchgap.services.verification_service import verify_gaps

ABSTRACT = (
    "This fictional paper evaluates limitations in academic systems "
    "using a controlled follow-up experiment with multiple data sources."
)


class OfflineOpenAlex:
    def search(self, query, limit=5):
        return [LiteratureWork(
            openalex_id="W10000000",
            title="FICTIONAL study of limitations in academic systems",
            url="https://openalex.org/W10000000", publication_year=2024,
            abstract=ABSTRACT,
        )]


class OfflineReviewModel:
    model = "FICTIONAL-NO-EXTERNAL-LLM"

    def extract_json(self, system, user):
        context = json.loads(user)
        return {"assessments": [
            {"openalex_id": w["openalex_id"], "relevance": "potentially_addresses",
             "reasoning": "Offline demonstration: fictional abstract overlaps the simulated question.",
             "evidence_quote": ABSTRACT}
            for w in context["works"]
        ]}


def run():
    extracted, analyzed, compared, gaps = run_four_phases()
    verified = verify_gaps(gaps, GapVerifier(OfflineReviewModel(), OfflineOpenAlex()))
    assert verified.status == "complete"
    assert len(verified.verifications) == len(gaps.gaps) == 3
    assert all(x.status == "challenged" for x in verified.verifications)
    assert verified.gap_discovery == gaps
    assert verified.novelty_established is False
    for gap in verified.gap_discovery.gaps:
        for fact in gap.cited_facts:
            paper = next(p for p in extracted.papers if p.document_id == fact.document_id)
            page = paper.pages[fact.evidence.page_number - 1]
            assert page.text[fact.evidence.start_char:fact.evidence.end_char] == fact.evidence.quote
    return verified


def main():
    result = run()
    print("OFFLINE TEST: All papers, literature works, and LLM replies are FICTIONAL")
    print(f"Phase 5: {len(result.verifications)} gap assessments; status={result.status}")
    for r in result.verifications:
        print(f"  {r.gap_id}: {r.status}, {r.searches_succeeded} simulated searches")
    print("All cited source-PDF offsets and OpenAlex abstract quotes passed checks")
    print("NO real research novelty or live OpenAlex/Groq operation was verified")


if __name__ == "__main__":
    main()
