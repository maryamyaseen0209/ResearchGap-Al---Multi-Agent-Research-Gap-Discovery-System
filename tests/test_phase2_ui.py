"""Headless end-to-end streamlit integration test for new AI button."""
import json

from researchgap.agents import paper_analyzer
from researchgap.core import analysis_config
from researchgap.integrations import groq_client
from researchgap.services import analysis_service
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub


class FakeLLMClient:
    model = "UI-test-model"

    def __init__(self, api_key, model):
        pass

    def extract_json(self, system, user):
        chunk = json.loads(user)["chunks"][0]
        return {"facts": [{"field": "finding", "text": "A testable outcome described in the research paper.",
                           "chunk_id": chunk["chunk_id"], "quote": chunk["text"][:45]}]}


def test_ui_full_phase1_to_phase2_flow(three_papers, monkeypatch):
    from researchgap import core
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(groq_client, "GroqClient", FakeLLMClient)
    # Since app.py imports names directly, these module monkeypatches apply when runpy executes it.
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 2
    phase1, phase2 = (json.loads(x) for x in stub.downloads)
    assert len(phase1["papers"]) == 3
    assert len(phase2["analyses"]) == 3
    assert all(x["status"] == "complete" for x in phase2["analyses"])
    assert all(x["findings"][0]["evidence"]["quote"] for x in phase2["analyses"])
