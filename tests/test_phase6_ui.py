"""Phase 6 headless Streamlit integration; model/search & graph are simulated."""

import json
from pathlib import Path

from langgraph_test_double import install
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub
from test_workflow_service import EmptySearch, ScientificFakeLLM
from researchgap.core import analysis_config
from researchgap.integrations import groq_client, openalex_client


class UIModel(ScientificFakeLLM):
    def __init__(self, api_key, model="fake-llm"):
        super().__init__()
        self.model = model


class UISearch(EmptySearch):
    calls = 0

    def __init__(self, api_key=None):
        super().__init__()

    def search(self, query, limit=5):
        UISearch.calls += 1
        return []


def uploaded_sample_papers():
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    return [MockUploadedFile(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))]


def enable_all(monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "FAKE-KEY")
    monkeypatch.setattr(groq_client, "GroqClient", UIModel)
    monkeypatch.setattr(openalex_client, "OpenAlexClient", UISearch)
    UISearch.calls = 0


def test_one_click_ui_generates_all_outputs(monkeypatch):
    enable_all(monkeypatch)
    ui = MockStreamlit(uploaded_sample_papers())
    ui.click_workflow = True
    load_app_with_stub(monkeypatch, ui)
    assert "workflow_result" in ui.session_state
    result = ui.session_state["workflow_result"]
    assert result.literature_verification is not None
    assert result.status == "complete"
    assert len(result.events) == 5
    assert UISearch.calls > 0
    final = json.loads(ui.downloads[-1])
    assert final["research_gaps"]["gaps"]
    assert final["initial_literature_verification"]
    assert final["novelty_established"] is False
    assert not ui.errors


def test_not_clicked_leaves_graph_uninitialized(monkeypatch):
    enable_all(monkeypatch)
    ui = MockStreamlit(uploaded_sample_papers())
    ui.click_workflow = False
    load_app_with_stub(monkeypatch, ui)
    assert "workflow_result" not in ui.session_state
    assert UISearch.calls == 0
    assert not ui.errors


def test_missing_api_key_prevents_one_click(monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: None)
    ui = MockStreamlit(uploaded_sample_papers())
    ui.click_workflow = True
    load_app_with_stub(monkeypatch, ui)
    assert "workflow_result" not in ui.session_state
    assert len(ui.downloads) == 1
    assert not ui.errors
