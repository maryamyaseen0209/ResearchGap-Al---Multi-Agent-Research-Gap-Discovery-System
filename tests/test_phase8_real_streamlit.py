"""Actual Streamlit AppTest smoke. Runs in CI/local after installing Streamlit."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="Streamlit is unavailable locally; CI installs it")
def test_real_streamlit_renders_workspace_without_user_uploads(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    project = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(project / "app.py"), default_timeout=15).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 8
    expected_tabs = [
        "01 · Library", "02 · Analyze", "03 · Compare", "04 · Discover",
        "05 · Verify", "06 · Agent workflow", "07 · Reports", "08 · Evaluation",
    ]
    assert [tab.label for tab in at.tabs] == expected_tabs
    assert any("ResearchGap" in str(m.value) for m in at.markdown)
    # Confirm initial UI states: process button is disabled without files
    assert at.button[0].disabled is True
    assert at.button[0].label == "Process research papers"
    # Verify informative guidance in individual tabs before uploads
    info_texts = [str(info.value) for info in at.info]
    assert any("Library tab" in text for text in info_texts)
