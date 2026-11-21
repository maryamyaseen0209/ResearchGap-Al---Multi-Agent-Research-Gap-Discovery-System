"""Headless widget wiring smoke test for environments without Streamlit.

This is not a replacement for a live-browser test with Streamlit installed.
"""

from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys
import types


class SessionDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class MockUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getvalue(self):
        return self.data


class MockStreamlit(types.ModuleType):
    def __init__(self, uploads):
        super().__init__("streamlit")
        self.uploads = uploads
        self.session_state = SessionDict()
        self.downloads = []
        self.errors = []
        self.warnings = []
        self.info_messages = []
        self.button_disabled = None
        self.rendered_pages = []
        self.rendered_chunks = []
        self.click_analyze = False
        self.click_comparison = False
        self.click_gaps = False
        self.click_verification = False
        self.click_workflow = False
        self.tables = []
        self.click_report = False
        self.click_baseline = False
        self.markdown_calls = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def markdown(self, text, *args, **kwargs):
        self.markdown_calls.append(text)

    def tabs(self, names):
        assert len(names) == 8
        return [self for _ in names]

    def text_input(self, label, value="", **kwargs):
        return value

    def metric(self, *args, **kwargs):
        pass

    def set_page_config(self, **kwargs):
        pass

    def title(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def divider(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def error(self, message, *args, **kwargs):
        self.errors.append(message)

    def info(self, message, *args, **kwargs):
        self.info_messages.append(message)

    def file_uploader(self, *args, **kwargs):
        assert kwargs['accept_multiple_files'] is True
        assert kwargs['type'] == ['pdf']
        return self.uploads

    def button(self, *args, **kwargs):
        if 'disabled' in kwargs:
            self.button_disabled = kwargs['disabled']
        if args and args[0] == "Analyze uploaded papers with AI" and not self.click_analyze:
            return False
        if args and args[0] == "Compare analyzed papers" and not self.click_comparison:
            return False
        if args and args[0] == "Discover candidate research gaps" and not self.click_gaps:
            return False
        if args and args[0] == "Verify candidate gaps against OpenAlex" and not self.click_verification:
            return False
        if args and args[0] == "Run full multi-agent workflow" and not self.click_workflow:
            return False
        if args and args[0] == "Generate research report" and not self.click_report:
            return False
        if args and args[0] == "Run single-pass evaluation baseline" and not self.click_baseline:
            return False
        return not self.button_disabled

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def expander(self, *args, **kwargs):
        return nullcontext()

    def columns(self, count):
        return [self for _ in range(count)]

    def download_button(self, label, data, **kwargs):
        self.downloads.append(data)

    def selectbox(self, label, options, **kwargs):
        return list(options)[0]

    def text_area(self, *args, **kwargs):
        self.rendered_pages.append(kwargs.get('value'))

    def table(self, value, *args, **kwargs):
        self.tables.append(value)

    def code(self, value, *args, **kwargs):
        self.rendered_chunks.append(value)


def load_app_with_stub(monkeypatch, stub):
    monkeypatch.setitem(sys.modules, 'streamlit', stub)
    runpy.run_path(str(Path(__file__).resolve().parent.parent / 'app.py'), run_name='__main__')


def test_ui_processes_three_pdfs_and_downloads_json(three_papers, monkeypatch):
    stub = MockStreamlit([MockUploadedFile(name, content) for name, content in three_papers])
    load_app_with_stub(monkeypatch, stub)
    assert stub.button_disabled is False
    assert len(stub.downloads) == 1
    payload = json.loads(stub.downloads[0])
    assert len(payload['papers']) == 3
    assert len(stub.rendered_pages) == 3
    assert len(stub.rendered_chunks) == 3
    assert not stub.errors


def test_ui_rejects_fewer_than_three_documents(three_papers, monkeypatch):
    name, content = three_papers[0]
    stub = MockStreamlit([MockUploadedFile(name, content)])
    load_app_with_stub(monkeypatch, stub)
    assert stub.button_disabled is True
    assert len(stub.downloads) == 0
    assert stub.info_messages


def test_ui_shows_partial_errors(three_papers, monkeypatch):
    uploads = [MockUploadedFile(n, d) for n, d in three_papers[:2]]
    uploads.append(MockUploadedFile('invalid.pdf', b'bad file'))
    stub = MockStreamlit(uploads)
    load_app_with_stub(monkeypatch, stub)
    assert stub.errors
    assert stub.warnings
    assert len(stub.downloads) == 1
