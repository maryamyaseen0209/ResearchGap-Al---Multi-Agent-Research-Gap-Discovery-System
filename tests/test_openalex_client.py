"""Phase 5 OpenAlex transport, metadata reconstruction and bounded failures."""

import requests
import pytest

from researchgap.integrations.openalex_client import (
    OpenAlexClient, LiteratureSearchError, parse_work, reconstruct_abstract,
)


def work_record(identifier="W123", title="Research on multilingual education"):
    return {
        "id": f"https://openalex.org/{identifier}",
        "title": title,
        "publication_year": 2024,
        "doi": "https://doi.org/10.1234/a_valid-doi",
        "abstract_inverted_index": {
            "This": [0], "paper": [1], "evaluates": [2],
            "multilingual": [3], "educational": [4], "models.": [5],
        },
        "cited_by_count": 3,
    }


class FakeResponse:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code = code
        self.body = body if body is not None else {"results": [work_record()]}
        self.headers = headers or {}

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_abstract_inverted_index_reconstruction():
    assert reconstruct_abstract({"world": [1], "Hello": [0]}) == "Hello world"
    assert reconstruct_abstract({"data": [0, 2], "are": [1]}) == "data are data"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract(["incorrect"]) is None
    assert reconstruct_abstract({"a": [1000]}) is None
    assert reconstruct_abstract({"a": [True, -1]}) is None


def test_safe_work_parser():
    work = parse_work(work_record())
    assert work.openalex_id == "W123"
    assert work.url == "https://openalex.org/W123"
    assert work.abstract == "This paper evaluates multilingual educational models."
    assert work.doi.startswith("https://doi.org/10.")
    assert parse_work({"id": "javascript:alert(1)", "title": "x"}) is None
    assert parse_work({"id": "https://attacker.example/W12", "title": "x"}) is None
    assert parse_work({"id": "https://openalex.org/W12", "title": ""}) is None
    assert parse_work(0) is None
    bad = work_record()
    bad["doi"] = "http://evil.example/not-a-doi"
    bad["publication_year"] = "2024"
    bad["cited_by_count"] = -10
    safe = parse_work(bad)
    assert safe.doi is None and safe.publication_year is None and safe.cited_by_count is None


def test_select_query_and_optional_key():
    session = FakeSession(FakeResponse(body={"results": [work_record(), work_record()]}))
    client = OpenAlexClient(api_key="test-key", session=session)
    works = client.search("multilingual educational models", limit=3)
    assert [w.openalex_id for w in works] == ["W123"]
    url, kwargs = session.calls[0]
    assert url.endswith("/works")
    assert kwargs["params"]["search.title_abstract_keywords"] == "multilingual educational models"
    assert kwargs["params"]["per_page"] == 3
    assert kwargs["params"]["api_key"] == "test-key"
    assert "abstract_inverted_index" in kwargs["params"]["select"]
    no_key_session = FakeSession(FakeResponse())
    OpenAlexClient(session=no_key_session).search("education")
    assert "api_key" not in no_key_session.calls[0][1]["params"]


def test_invalid_input_rejected():
    client = OpenAlexClient(session=FakeSession())
    for bad in ("", "  ", "z" * 181):
        with pytest.raises(ValueError):
            client.search(bad)
    for limit in (0, 16):
        with pytest.raises(ValueError):
            client.search("valid", limit=limit)
    with pytest.raises(ValueError):
        OpenAlexClient(timeout_seconds=0)
    with pytest.raises(ValueError):
        OpenAlexClient(max_retries=5)


def test_rate_limit_then_success(monkeypatch):
    monkeypatch.setattr("researchgap.integrations.openalex_client.time.sleep", lambda seconds: None)
    session = FakeSession(FakeResponse(429, headers={"Retry-After": "200"}), FakeResponse())
    assert OpenAlexClient(session=session).search("test")
    assert len(session.calls) == 2


@pytest.mark.parametrize("code,expected", [
    (401, "AUTH_ERROR"), (403, "AUTH_ERROR"), (429, "RATE_LIMITED"),
    (404, "API_ERROR"), (503, "API_ERROR"),
])
def test_http_errors(code, expected):
    with pytest.raises(LiteratureSearchError) as exc:
        OpenAlexClient(session=FakeSession(FakeResponse(code)), max_retries=0).search("test")
    assert exc.value.code == expected


def test_retry_network_error_then_success(monkeypatch):
    monkeypatch.setattr("researchgap.integrations.openalex_client.time.sleep", lambda seconds: None)
    session = FakeSession(requests.Timeout(), FakeResponse())
    assert OpenAlexClient(session=session).search("test")


def test_network_error_on_final_retry():
    with pytest.raises(LiteratureSearchError) as exc:
        OpenAlexClient(session=FakeSession(requests.Timeout()), max_retries=0).search("test")
    assert exc.value.code == "NETWORK_ERROR"


@pytest.mark.parametrize("body", [[], {}, {"results": None}, ValueError("invalid JSON")])
def test_bad_json_or_schema_is_safe(body):
    with pytest.raises(LiteratureSearchError) as exc:
        OpenAlexClient(session=FakeSession(FakeResponse(body=body)), max_retries=0).search("test")
    assert exc.value.code == "BAD_RESPONSE"


def test_bad_or_duplicate_records_ignored():
    response = FakeResponse(body={"results": [
        {"id": "https://evil.org/W55", "title": "fake"},
        work_record("W123"), work_record("W123"), work_record("W124", "Other research"),
    ]})
    result = OpenAlexClient(session=FakeSession(response)).search("test", limit=4)
    assert [w.openalex_id for w in result] == ["W123", "W124"]


def test_all_invalid_records_do_not_look_like_no_search_matches():
    bad = FakeResponse(body={"results": [{"id": "javascript:evil()", "title": "false"}]})
    with pytest.raises(LiteratureSearchError) as exc:
        OpenAlexClient(session=FakeSession(bad)).search("test")
    assert exc.value.code == "BAD_RESPONSE"
