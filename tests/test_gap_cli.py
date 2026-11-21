"""Offline, reproducible Phase 4 CLI tests."""

import json

import pytest

from researchgap import gap_cli
from researchgap.core import analysis_config
from test_gap_discoverer import FakeGapLLM, input_comparison  # noqa: F401


def test_cli_processes_phase3_export(input_comparison, tmp_path, monkeypatch):
    source = tmp_path / "comparison.json"
    source.write_text(input_comparison.model_dump_json(), encoding="utf8")
    destination = tmp_path / "reports" / "gaps.json"
    monkeypatch.setattr(analysis_config, "configured_api_key", lambda: "test-key")
    monkeypatch.setattr(gap_cli, "configured_api_key", lambda: "test-key")
    monkeypatch.setattr(gap_cli, "GroqClient", lambda api_key, model: FakeGapLLM())
    code = gap_cli.main([str(source), "--output", str(destination)])
    assert code == 0
    data = json.loads(destination.read_text(encoding="utf8"))
    assert len(data["gaps"]) == 4
    assert data["literature_verification"] == "not_performed"
    assert all(g["cited_facts"] for g in data["gaps"])


def test_cli_missing_key(input_comparison, tmp_path, monkeypatch):
    source = tmp_path / "comparison.json"
    source.write_text(input_comparison.model_dump_json(), encoding="utf8")
    monkeypatch.setattr(gap_cli, "configured_api_key", lambda: None)
    with pytest.raises(SystemExit) as err:
        gap_cli.main([str(source)])
    assert err.value.code == 2


def test_cli_rejects_unreadable_json(tmp_path, monkeypatch):
    source = tmp_path / "bad.json"
    source.write_text("not JSON", encoding="utf8")
    monkeypatch.setattr(gap_cli, "configured_api_key", lambda: "test-key")
    with pytest.raises(SystemExit) as err:
        gap_cli.main([str(source)])
    assert err.value.code == 2


def test_cli_returns_nonzero_for_insufficient_evidence(input_comparison, tmp_path, monkeypatch):
    for row in input_comparison.comparison_table:
        row.limitations = []
    source = tmp_path / "comparison.json"
    source.write_text(input_comparison.model_dump_json(), encoding="utf8")
    destination = tmp_path / "gaps.json"
    monkeypatch.setattr(gap_cli, "configured_api_key", lambda: "test-key")
    monkeypatch.setattr(gap_cli, "GroqClient", lambda api_key, model: FakeGapLLM())
    code = gap_cli.main([str(source), "--output", str(destination)])
    assert code == 1
    assert json.loads(destination.read_text(encoding="utf8"))["gaps"] == []
