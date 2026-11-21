"""Phase 5 verification tests with FICTIONAL papers and fake APIs/models."""

import json
from copy import deepcopy

import pytest

from researchgap.agents.gap_verifier import GapVerifier, quote_matches, search_queries
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.agents.paper_analyzer import PaperAnalyzer
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.core.verification_config import VerificationSettings
from researchgap.core.verification_schemas import LiteratureVerificationResult, LiteratureWork
from researchgap.integrations.groq_client import LLMError
from researchgap.integrations.openalex_client import LiteratureSearchError
from researchgap.services.analysis_service import analyze_batch
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.services.comparison_service import compare_batch
from researchgap.services.gap_service import discover_gaps
from researchgap.services.verification_service import GapVerificationNode, verify_gaps
from test_gap_discoverer import FakeEvidenceExtractor, FakeComparisonLLM, FakeGapLLM

ABSTRACT = "Our study evaluates the reported limitations across academic systems using controlled experiments."


def mk_work(work_id="W100", abstract=ABSTRACT):
    return LiteratureWork(
        openalex_id=work_id, title="Fictional academic systems evaluation",
        url=f"https://openalex.org/{work_id}",
        publication_year=2025, abstract=abstract,
    )


class FakeSearch:
    def __init__(self, works=None, failures=None):
        self.works = works if works is not None else [mk_work()]
        self.failures = failures or set()
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        if len(self.calls) in self.failures:
            raise LiteratureSearchError("RATE_LIMITED", "fictional simulated rate limit")
        return self.works[:limit]


class FakeReviewer:
    model = "fake-reviewer"

    def __init__(self, behavior="direct"):
        self.behavior = behavior
        self.calls = []

    def extract_json(self, system_message, user_message):
        self.calls.append((system_message, user_message))
        if self.behavior == "error":
            raise LLMError("RATE_LIMITED", "fictional LLM failure")
        if self.behavior == "broken":
            return {"unexpected": []}
        works = json.loads(user_message)["works"]
        results = []
        for work in works:
            relevance = "potentially_addresses" if self.behavior in (
                "direct", "fabricated_quote", "title_only", "duplicate"
            ) else "related" if self.behavior == "related" else "uncertain"
            quote = (ABSTRACT if self.behavior in ("direct", "duplicate") else
                     "This is an invented quotation that does not appear in the source." if self.behavior == "fabricated_quote" else
                     "Fictional academic systems evaluation" if self.behavior == "title_only" else
                     "Fictional academic systems evaluation" if self.behavior == "related" else None)
            results.append({
                "openalex_id": work["openalex_id"], "relevance": relevance,
                "evidence_quote": quote,
                "reasoning": "This assessment is a fictional, source-referenced simulated judgement.",
            })
        if self.behavior == "unknown_id":
            results[0]["openalex_id"] = "W999999"
        if self.behavior == "missing":
            results = []
        if self.behavior == "duplicate":
            results.append(results[0])
        return {"assessments": results}


@pytest.fixture
def discovery(three_papers):
    batch = process_batch([PaperUpload(name, content) for name, content in three_papers])
    analyses = analyze_batch(batch, PaperAnalyzer(FakeEvidenceExtractor()))
    comparison = compare_batch(analyses, PaperComparator(FakeComparisonLLM()))
    result = discover_gaps(comparison, GapDiscoverer(FakeGapLLM()))
    assert result.gaps
    return result


def single_gap(discovery):
    result = discovery.model_copy(deep=True)
    result.gaps = result.gaps[:1]
    return result


def test_query_generation_is_bounded_and_reproducible(discovery):
    queries = search_queries(discovery.gaps[0])
    assert 1 <= len(queries) <= 2
    assert queries == search_queries(discovery.gaps[0])
    assert all(len(q) <= 140 for q in queries)
    assert not any(q.startswith("How") for q in queries)
    assert quote_matches(" academic   systems ", "Across Academic Systems in practice")
    assert not quote_matches("invented", ABSTRACT)
    assert search_queries(discovery.gaps[0], limit=1) == queries[:1]


def test_direct_abstract_evidence_challenges_gap(discovery):
    result = verify_gaps(single_gap(discovery), GapVerifier(FakeReviewer(), FakeSearch()))
    assert result.status == "complete"
    assert result.verifications[0].status == "challenged"
    assert result.verifications[0].assessments[0].evidence_quote == ABSTRACT
    assert result.verifications[0].works[0].url == "https://openalex.org/W100"
    assert result.novelty_established is False
    assert result.human_review_required is True
    serialized = result.model_dump_json()
    assert LiteratureVerificationResult.model_validate_json(serialized) == result
    assert result.gap_discovery == single_gap(discovery)


def test_no_direct_match_only_with_successful_queries(discovery):
    result = GapVerifier(FakeReviewer("related"), FakeSearch()).verify(single_gap(discovery))
    one = result.verifications[0]
    assert one.status == "no_direct_match_found"
    assert one.searches_succeeded == one.searches_attempted
    assert any("does NOT establish novelty" in w for w in one.warnings)


def test_empty_results_does_not_prove_novelty(discovery):
    result = GapVerifier(FakeReviewer("direct"), FakeSearch(works=[])).verify(single_gap(discovery))
    assert result.verifications[0].status == "no_direct_match_found"
    assert not result.verifications[0].assessments
    assert result.verifications[0].works_retrieved == 0
    assert result.novelty_established is False


@pytest.mark.parametrize("behavior", ["fabricated_quote", "title_only", "unknown_id", "missing", "broken", "error"])
def test_invalid_or_missing_assessment_is_inconclusive(discovery, behavior):
    result = GapVerifier(FakeReviewer(behavior), FakeSearch()).verify(single_gap(discovery))
    assert result.status == "partial"
    assert result.verifications[0].status == "inconclusive"
    assert not any(a.relevance == "potentially_addresses" for a in result.verifications[0].assessments)


def test_incomplete_search_does_not_claim_no_direct_match(discovery):
    result = GapVerifier(FakeReviewer("related"), FakeSearch(failures={2})).verify(single_gap(discovery))
    one = result.verifications[0]
    assert one.status == "inconclusive"
    assert one.searches_attempted == 2
    assert one.searches_succeeded == 1
    assert result.status == "partial"


def test_all_searches_failed_reports_unavailable(discovery):
    result = GapVerifier(FakeReviewer(), FakeSearch(failures={1, 2})).verify(single_gap(discovery))
    assert result.status == "unavailable"
    assert result.verifications[0].status == "inconclusive"
    assert result.verifications[0].works_retrieved == 0


def test_works_without_abstract_cannot_exclude_direct_match(discovery):
    result = GapVerifier(FakeReviewer("related"), FakeSearch(works=[mk_work(abstract=None)])).verify(single_gap(discovery))
    assert result.verifications[0].status == "inconclusive"
    assert result.verifications[0].assessments[0].relevance == "uncertain"


def test_valid_related_quote_and_duplicate_id_safe(discovery):
    result = GapVerifier(FakeReviewer("duplicate"), FakeSearch(works=[mk_work(), mk_work()])).verify(single_gap(discovery))
    assert result.verifications[0].status == "challenged"
    assert len(result.verifications[0].works) == 1


def test_multiple_gaps_and_no_gaps(discovery):
    searcher = FakeSearch(works=[])
    result = GapVerifier(FakeReviewer(), searcher).verify(discovery)
    assert len(result.verifications) == len(discovery.gaps)
    assert result.status == "complete"
    empty = deepcopy(discovery)
    empty.gaps = []
    before = len(searcher.calls)
    result = GapVerifier(FakeReviewer(), searcher).verify(empty)
    assert result.status == "no_gaps"
    assert result.verifications == []
    assert len(searcher.calls) == before


def test_node_accepts_dict_state(discovery):
    node = GapVerificationNode(GapVerifier(FakeReviewer(), FakeSearch(works=[])))
    output = node({"research_gaps": discovery.model_dump(mode="json")})
    assert output["literature_verification"]["source"] == "OpenAlex"
    assert output["literature_verification"]["novelty_established"] is False


def test_context_limit_keeps_results_inconclusive(discovery):
    short_settings = VerificationSettings(max_context_characters=3000)
    huge = mk_work()
    huge.abstract = ABSTRACT + " Research question. " * 2000
    result = GapVerifier(FakeReviewer("related"), FakeSearch(works=[huge]), settings=short_settings).verify(single_gap(discovery))
    assert result.verifications[0].status in ("inconclusive", "no_direct_match_found")
    # Even when review fits, a 'no match' is only about these results, not global novelty.
    assert result.novelty_established is False


def test_verifier_does_not_change_phase4_gap_evidence(discovery):
    old = discovery.model_dump_json()
    GapVerifier(FakeReviewer(), FakeSearch()).verify(single_gap(discovery))
    assert discovery.model_dump_json() == old


def test_challenge_from_partial_search_is_still_partial_overall(discovery):
    report = GapVerifier(FakeReviewer("direct"), FakeSearch(failures={2})).verify(single_gap(discovery))
    assert report.verifications[0].status == "challenged"
    assert report.status == "partial"


def test_reviewer_related_without_quote_is_not_trusted(discovery):
    class NoQuote:
        model = "fake"

        def extract_json(self, system, user):
            return {"assessments": [{
                "openalex_id": w["openalex_id"], "relevance": "related",
                "reasoning": "The fictional sample may be related to the broad topic.",
                "evidence_quote": None,
            } for w in json.loads(user)["works"]]}

    report = GapVerifier(NoQuote(), FakeSearch()).verify(single_gap(discovery))
    assert report.verifications[0].status == "inconclusive"


def test_no_meaningful_search_query_reports_inconclusive(discovery):
    report_input = single_gap(discovery)
    report_input.gaps[0].title = "The research study is about a candidate gap"
    report_input.gaps[0].research_questions[0].text = "How does the research paper study the research gap?"
    searcher = FakeSearch()
    report = GapVerifier(FakeReviewer(), searcher).verify(report_input)
    assert report.verifications[0].searches_attempted == 0
    assert report.verifications[0].status == "inconclusive"
    assert not searcher.calls


def test_verification_settings_validate_limits():
    with pytest.raises(ValueError):
        VerificationSettings(max_works_per_gap=2, max_results_per_query=5)
