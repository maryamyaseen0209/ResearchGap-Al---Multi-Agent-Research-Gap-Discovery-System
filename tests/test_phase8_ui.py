"""Phase 8 Evaluation Lab is explicit and has correct source invalidation."""

from pathlib import Path
from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub
from test_phase6_ui import enable_all
from researchgap.ui.dashboard import render_evaluation_studio
from researchgap.services.batch_processor import PaperUpload, process_batch
from examples.demo_phase8_offline import OfflineBaselineModel, execute
from researchgap.core.workflow_schemas import WorkflowResult
from researchgap.evaluation.baseline import SinglePassBaseline
from researchgap.evaluation.metrics import compare_systems


def _uploads():
    folder = Path(__file__).resolve().parent.parent / "examples" / "sample_papers"
    return [MockUploadedFile(p.name, p.read_bytes()) for p in sorted(folder.glob("*.pdf"))]


def test_8th_workspace_tab_no_inference_without_button(monkeypatch):
    enable_all(monkeypatch)
    ui = MockStreamlit(_uploads())
    ui.click_workflow = True
    ui.click_baseline = False
    load_app_with_stub(monkeypatch, ui)
    assert "workflow_result" in ui.session_state
    assert "baseline_result" not in ui.session_state
    assert "evaluation_result" not in ui.session_state
    assert not ui.errors


def test_evaluation_studio_generates_three_portable_downloads(tmp_path, monkeypatch):
    # Put a fictional fully populated workflow into session state; no live API.
    out = tmp_path / "smoke"
    execute(out)
    batch = process_batch([PaperUpload(f.name, f.data) for f in _uploads()])
    workflow = WorkflowResult.model_validate_json((out / "workflow_results.json").read_text(encoding="utf-8"))
    monkeypatch.setattr("researchgap.core.analysis_config.configured_api_key", lambda: None)
    ui = MockStreamlit([])
    ui.session_state["processing_result"] = batch
    ui.session_state["workflow_result"] = workflow
    ui.session_state["baseline_result"] = SinglePassBaseline(OfflineBaselineModel()).run(workflow.paper_analyses)
    ui.session_state["evaluation_result"] = compare_systems(batch, ui.session_state["baseline_result"], workflow)
    ui.session_state["evaluation_source_signature"] = (id(batch), id(workflow))
    render_evaluation_studio(ui)
    assert len(ui.downloads) == 3
    assert b"citation" in ui.downloads[0].lower()
    assert b"supported_by_evidence" in ui.downloads[1]
    assert b"baseline" in ui.downloads[2]
    assert not ui.errors


def test_evaluation_outputs_invalidated_when_workflow_changes(tmp_path, monkeypatch):
    out = tmp_path / "smoke"
    execute(out)
    batch = process_batch([PaperUpload(f.name, f.data) for f in _uploads()])
    first = WorkflowResult.model_validate_json((out / "workflow_results.json").read_text(encoding="utf-8"))
    second = first.model_copy(update={"run_id": "changed"})
    monkeypatch.setattr("researchgap.core.analysis_config.configured_api_key", lambda: None)
    ui = MockStreamlit([])
    ui.session_state["processing_result"] = batch
    ui.session_state["workflow_result"] = first
    ui.session_state["baseline_result"] = SinglePassBaseline(OfflineBaselineModel()).run(first.paper_analyses)
    ui.session_state["evaluation_result"] = compare_systems(batch, ui.session_state["baseline_result"], first)
    ui.session_state["evaluation_source_signature"] = (id(batch), id(first))
    ui.session_state["workflow_result"] = second
    render_evaluation_studio(ui)
    assert "baseline_result" not in ui.session_state
    assert "evaluation_result" not in ui.session_state
    assert not ui.downloads
