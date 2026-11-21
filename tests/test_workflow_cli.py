"""Unified Phase 6 CLI tests (no external services, no real LangGraph)."""

import json

import pytest

from researchgap import workflow_cli
from researchgap.integrations import groq_client, openalex_client
from researchgap.core import analysis_config
from langgraph_test_double import install
from test_workflow_service import EmptySearch, ScientificFakeLLM
from researchgap.services.batch_processor import PaperUpload, process_batch


class CLIModel(ScientificFakeLLM):
    def __init__(self, api_key, model="fake-llm"):
        super().__init__()
        self.model = model


class CLISearch(EmptySearch):
    def __init__(self, api_key=None):
        super().__init__()


def test_cli_export_round_trip(monkeypatch, tmp_path):
    install(monkeypatch)
    monkeypatch.setattr(workflow_cli, "configured_api_key", lambda: "fake")
    monkeypatch.setattr(workflow_cli, "GroqClient", CLIModel)
    monkeypatch.setattr(workflow_cli, "OpenAlexClient", CLISearch)
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    batch = process_batch([PaperUpload(p.name, p.read_bytes()) for p in sorted(sample.glob("*.pdf"))])
    source = tmp_path / "input.json"
    source.write_text(batch.model_dump_json(), encoding="utf-8")
    out = tmp_path / "nested" / "workflow.json"
    assert workflow_cli.main([str(source), "--output", str(out), "--no-revision"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["literature_verification"]["status"] == "complete"
    assert saved["revision_count"] == 0
    assert saved["novelty_established"] is False


def test_cli_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow_cli, "configured_api_key", lambda: None)
    with pytest.raises(SystemExit) as exc:
        workflow_cli.main([str(tmp_path / "nonexistent.json")])
    assert exc.value.code == 2


def test_cli_rejects_invalid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow_cli, "configured_api_key", lambda: "FAKE")
    source = tmp_path / "bad.json"
    source.write_text("not-json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        workflow_cli.main([str(source)])
    assert exc.value.code == 2
