"""Phase 7 report studio and new workspace tab interactions, all headless."""

import json

from test_streamlit_wiring import MockStreamlit, MockUploadedFile, load_app_with_stub


def test_new_workspace_renders_seven_tabs_and_not_report_until_requested(three_papers, monkeypatch):
    ui = MockStreamlit([MockUploadedFile(name, raw) for name, raw in three_papers])
    load_app_with_stub(monkeypatch, ui)
    assert "report_artifacts" not in ui.session_state
    assert len(ui.downloads) == 1
    assert any("rg-hero" in text for text in ui.markdown_calls)
    assert any("rg-steps" in text for text in ui.markdown_calls)
    assert not ui.errors


def test_generate_report_button_exports_three_formats(three_papers, monkeypatch):
    ui = MockStreamlit([MockUploadedFile(name, raw) for name, raw in three_papers])
    ui.click_report = True
    load_app_with_stub(monkeypatch, ui)
    assert "report_artifacts" in ui.session_state
    # Original Phase 1 JSON followed by Phase 7 Markdown, HTML and JSON exports.
    assert len(ui.downloads) == 4
    assert ui.downloads[1].startswith(b"# ")
    assert b"<html" in ui.downloads[2]
    report = json.loads(ui.downloads[3])
    assert report["report_metadata"]["novelty_established"] is False
    assert len(report["processing"]["papers"]) == 3
    assert not ui.errors


def test_report_stays_unavailable_without_any_papers(monkeypatch):
    ui = MockStreamlit([])
    ui.click_report = True
    load_app_with_stub(monkeypatch, ui)
    assert "report_artifacts" not in ui.session_state
    assert len(ui.downloads) == 0
    assert ui.info_messages


def test_report_invalidated_when_source_batch_changes(three_papers):
    from researchgap.services.batch_processor import PaperUpload, process_batch
    from researchgap.ui.dashboard import render_report_studio
    ui = MockStreamlit([])
    ui.click_report = True
    first = process_batch([PaperUpload(name, content) for name, content in three_papers])
    ui.session_state["processing_result"] = first
    render_report_studio(ui)
    assert "report_artifacts" in ui.session_state
    assert len(ui.downloads) == 3

    second = process_batch([PaperUpload(name, content) for name, content in reversed(three_papers)])
    ui.session_state["processing_result"] = second
    ui.click_report = False
    render_report_studio(ui)
    assert "report_artifacts" not in ui.session_state
    assert len(ui.downloads) == 3  # No stale second report downloaded
