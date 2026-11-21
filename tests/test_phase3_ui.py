"""End-to-end mocked Streamlit path: PDFs -> Phase 2 -> Phase 3 comparison."""

import json

from researchgap.core import analysis_config
from researchgap.integrations import groq_client
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub


class UniversalFakeClient:
    model = "fake-ui-model"

    def __init__(self, api_key, model):
        self.model = model

    def extract_json(self, system, user):
        payload = json.loads(user)
        if "chunks" in payload:
            chunk = payload["chunks"][0]
            return {"facts": [{
                "field": "methodology", "text": "Source describes study methods and approach.",
                "chunk_id": chunk["chunk_id"], "quote": chunk["text"][:45],
            }]}
        fact_ids = [x["fact_id"] for x in payload["facts"] if x["category"] == "methodology"]
        return {"insights": [{
            "kind": "method_similarity",
            "statement": "The selected papers both describe research methods in the available text.",
            "fact_ids": fact_ids[:2],
        }]}


def test_full_ui_phase1_to_phase3(three_papers, monkeypatch):
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(groq_client, "GroqClient", UniversalFakeClient)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    stub.click_comparison = True
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 3
    phase1, phase2, phase3 = [json.loads(data) for data in stub.downloads]
    assert len(phase1["papers"]) == 3
    assert len(phase2["analyses"]) == 3
    assert phase3["paper_count"] == 3
    assert phase3["status"] == "complete"
    assert len(phase3["insights"]) == 1
    assert len(stub.tables) == 1
    assert len(stub.tables[0]) == 3
    assert not stub.errors
