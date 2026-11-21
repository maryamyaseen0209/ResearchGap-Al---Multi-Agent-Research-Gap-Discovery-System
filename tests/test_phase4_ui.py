"""Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 UI smoke test without Streamlit installed."""

import json

from researchgap.core import analysis_config
from researchgap.integrations import groq_client
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub
from test_gap_discoverer import make_candidate


class MultiPhaseFakeClient:
    model = "phase4-ui-simulation"

    def __init__(self, api_key, model):
        self.model = model

    def extract_json(self, system, user):
        data = json.loads(user)
        if "chunks" in data:
            c = data["chunks"][0]
            return {"facts": [
                {"field": field, "text": f"Research study examines {field} information in the source.",
                 "chunk_id": c["chunk_id"], "quote": c["text"][:60]}
                for field in ("limitation", "methodology", "dataset", "finding")
            ]}
        if "comparison_insights" in data:
            facts = data["facts"]
            limits = [f["fact_id"] for f in facts if f["category"] == "limitation"]
            methods = [f["fact_id"] for f in facts if f["category"] == "methodology"]
            datasets = [f["fact_id"] for f in facts if f["category"] == "dataset"]
            return {"gaps": [
                make_candidate("shared_limitation", limits[:2]),
                make_candidate("methodological_extension", [limits[0], methods[1]]),
                make_candidate("dataset_coverage", [limits[0], datasets[1]]),
            ]}
        limits = [f["fact_id"] for f in data["facts"] if f["category"] == "limitation"]
        return {"insights": [{
            "kind": "shared_limitation", "statement": "The two papers describe limitations in their studies.",
            "fact_ids": limits[:2],
        }]}


def test_full_ui_four_phases(three_papers, monkeypatch):
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(groq_client, "GroqClient", MultiPhaseFakeClient)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    stub.click_comparison = True
    stub.click_gaps = True
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 4
    phase1, phase2, phase3, phase4 = [json.loads(data) for data in stub.downloads]
    assert len(phase1["papers"]) == 3
    assert len(phase2["analyses"]) == 3
    assert len(phase3["insights"]) == 1
    assert len(phase4["gaps"]) == 3
    assert phase4["status"] == "complete"
    assert phase4["gaps"][0]["cited_facts"][0]["evidence"]["quote"]
    assert not stub.errors


def test_phase4_not_called_without_explicit_click(three_papers, monkeypatch):
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(groq_client, "GroqClient", MultiPhaseFakeClient)
    stub = MockStreamlit([MockUploadedFile(n, d) for n, d in three_papers])
    stub.click_analyze = True
    stub.click_comparison = True
    stub.click_gaps = False
    load_app_with_stub(monkeypatch, stub)
    assert len(stub.downloads) == 3
    assert "gap_result" not in stub.session_state
