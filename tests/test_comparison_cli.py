"""Phase 3 accepts the Phase 2 JSON export from disk, without PDF re-processing."""
import json

import pytest

from researchgap import comparison_cli
from test_paper_comparator import ComparisonFakeLLM, compared_analyses


class FakeClient(ComparisonFakeLLM):
    def __init__(self, api_key, model):
        super().__init__()
        self.model = model


def test_cli_reads_phase2_json_and_exports_phase3_result(tmp_path, compared_analyses, monkeypatch):
    source = tmp_path / "phase2.json"
    target = tmp_path / "phase3.json"
    source.write_text(compared_analyses.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(comparison_cli, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(comparison_cli, "GroqClient", FakeClient)
    assert comparison_cli.main([str(source), "--output", str(target)]) == 0
    result = json.loads(target.read_text(encoding="utf-8"))
    assert len(result["comparison_table"]) == 3
    assert len(result["insights"]) == 1
    assert result["insights"][0]["cited_facts"][0]["evidence"]["quote"]


def test_cli_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison_cli, "configured_api_key", lambda: None)
    with pytest.raises(SystemExit) as exc:
        comparison_cli.main([str(tmp_path / "unavailable.json")])
    assert exc.value.code == 2


def test_cli_rejects_incomplete_input(tmp_path, compared_analyses, monkeypatch):
    source = tmp_path / "phase2.json"
    source.write_text(compared_analyses.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(comparison_cli, "configured_api_key", lambda: "fake-key")
    monkeypatch.setattr(comparison_cli, "GroqClient", FakeClient)
    compared_analyses.analyses = compared_analyses.analyses[:2]
    source.write_text(compared_analyses.model_dump_json(), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        comparison_cli.main([str(source)])
    assert exc.value.code == 2
