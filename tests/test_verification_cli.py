"""Phase 5 CLI export, error and output-state regression tests."""

import json

import pytest

from examples.demo_phase4_offline import run as run_phase4
from researchgap import verification_cli
from test_gap_verifier import FakeReviewer, FakeSearch


class FakeCliLLM(FakeReviewer):
    def __init__(self, api_key=None, model="fake-cli"):
        super().__init__("direct")
        self.model = model


class FakeCliOpenAlex(FakeSearch):
    def __init__(self, api_key=None):
        super().__init__()


def test_phase5_cli_exports_comprehensive_json(monkeypatch, tmp_path, capsys):
    _, _, _, gaps = run_phase4()
    input_path = tmp_path / "candidate_gaps.json"
    output_path = tmp_path / "results" / "verification.json"
    input_path.write_text(gaps.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(verification_cli, "configured_api_key", lambda: "fake")
    monkeypatch.setattr(verification_cli, "GroqClient", FakeCliLLM)
    monkeypatch.setattr(verification_cli, "OpenAlexClient", FakeCliOpenAlex)
    assert verification_cli.main([str(input_path), "--output", str(output_path)]) == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["source"] == "OpenAlex"
    assert data["novelty_established"] is False
    assert len(data["verifications"]) == 3
    assert all(i["status"] == "challenged" for i in data["verifications"])
    assert data["gap_discovery"] == gaps.model_dump(mode="json")
    assert "cannot prove global novelty" in capsys.readouterr().out


def test_missing_key_exits_safely(monkeypatch, tmp_path):
    monkeypatch.setattr(verification_cli, "configured_api_key", lambda: None)
    with pytest.raises(SystemExit) as exc:
        verification_cli.main([str(tmp_path / "missing.json")])
    assert exc.value.code == 2


def test_invalid_input_does_not_write_output(monkeypatch, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"unsupported": true}', encoding="utf-8")
    monkeypatch.setattr(verification_cli, "configured_api_key", lambda: "fake")
    with pytest.raises(SystemExit) as exc:
        verification_cli.main([str(path), "--output", str(tmp_path / "out.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "out.json").exists()


def test_offline_five_phase_demo():
    from examples.demo_phase5_offline import run
    output = run()
    assert len(output.verifications) == 3
    assert output.status == "complete"
    assert all(x.works_reviewed == 1 for x in output.verifications)
