"""Headless five-phase UI flow. No live Streamlit, Groq or OpenAlex calls."""

import json

from researchgap.core import analysis_config
from researchgap.integrations import groq_client, openalex_client
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub
from test_phase4_ui import MultiPhaseFakeClient
from test_gap_verifier import mk_work, ABSTRACT


class FivePhaseLLM(MultiPhaseFakeClient):
    def extract_json(self, system, user):
        data = json.loads(user)
        if "candidate_gap" in data and "works" in data:
            return {"assessments": [
                {"openalex_id": work["openalex_id"], "relevance": "potentially_addresses",
                 "reasoning": "Simulated study appears to address this fictional research question.",
                 "evidence_quote": ABSTRACT}
                for work in data["works"]
            ]}
        return super().extract_json(system, user)


class FakeOpenAlex:
    calls = []

    def __init__(self, api_key=None):
        self.api_key = api_key

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return [mk_work()]


def install_fakes(monkeypatch):
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(groq_client, "GroqClient", FivePhaseLLM)
    monkeypatch.setattr(openalex_client, "OpenAlexClient", FakeOpenAlex)
    FakeOpenAlex.calls.clear()


def test_five_phase_ui_end_to_end(three_papers, monkeypatch):
    install_fakes(monkeypatch)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    stub.click_comparison = True
    stub.click_gaps = True
    stub.click_verification = True
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 5
    phase1, phase2, phase3, phase4, phase5 = (json.loads(x) for x in stub.downloads)
    assert len(phase1["papers"]) == len(phase2["analyses"]) == 3
    assert len(phase3["comparison_table"]) == 3
    assert len(phase4["gaps"]) == 3
    assert len(phase5["verifications"]) == 3
    assert all(x["status"] == "challenged" for x in phase5["verifications"])
    assert phase5["gap_discovery"] == phase4
    assert phase5["novelty_established"] is False
    assert len(FakeOpenAlex.calls) >= 3
    assert not stub.errors


def test_verification_never_runs_until_clicked(three_papers, monkeypatch):
    install_fakes(monkeypatch)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    stub.click_comparison = True
    stub.click_gaps = True
    stub.click_verification = False
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 4
    assert not FakeOpenAlex.calls
    assert "verification_result" not in stub.session_state


def test_verification_no_api_key(three_papers, monkeypatch):
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: None)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_verification = True
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 1
    assert "verification_result" not in stub.session_state
