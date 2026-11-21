import json
import pytest

from researchgap import analysis_cli
from researchgap.services.batch_processor import PaperUpload, process_batch


class FakeClient:
    model = "fake"

    def __init__(self, api_key, model):
        self.model = model

    def extract_json(self, system_message, user_message):
        chunk = json.loads(user_message)["chunks"][0]
        return {"facts": [{"field": "methodology", "text": "The authors describe an evaluation methodology.",
                           "chunk_id": chunk["chunk_id"], "quote": chunk["text"][:70]}]}


def test_cli_analyzes_extracted_phase1_json(tmp_path, three_papers, monkeypatch):
    source = process_batch([PaperUpload(name, content) for name, content in three_papers])
    input_path = tmp_path / "phase1.json"
    output_path = tmp_path / "phase2.json"
    input_path.write_text(source.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(analysis_cli, "configured_api_key", lambda: "mock-key")
    monkeypatch.setattr(analysis_cli, "GroqClient", FakeClient)
    assert analysis_cli.main([str(input_path), "--output", str(output_path)]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["analyses"]) == 3
    assert len(payload["analyses"][0]["methodologies"]) == 1
    assert payload["analyses"][0]["methodologies"][0]["evidence"]["start_char"] >= 0


def test_cli_requires_key(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_cli, "configured_api_key", lambda: None)
    with pytest.raises(SystemExit) as exc:
        analysis_cli.main([str(tmp_path / "phase1.json")])
    assert exc.value.code == 2
