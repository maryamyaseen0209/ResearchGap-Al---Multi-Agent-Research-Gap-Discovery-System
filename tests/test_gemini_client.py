"""Unit tests for Google Gemini REST API adapter."""
import json
import pytest
import requests

from researchgap.integrations.gemini_client import GeminiClient, GEMINI_BASE_URL
from researchgap.integrations.groq_client import LLMError


class FakeResponse:
    def __init__(self, status=200, value=None, headers=None, text=""):
        self.status_code = status
        self.value = value if value is not None else {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps({"facts": []})}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 42,
                "candidatesTokenCount": 18,
                "totalTokenCount": 60,
            },
        }
        self.headers = headers or {}
        self.text = text or json.dumps(self.value)

    def json(self):
        return self.value


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        res = next(self.responses)
        if isinstance(res, Exception):
            raise res
        return res


def test_valid_gemini_request_and_json():
    session = FakeSession([FakeResponse()])
    client = GeminiClient(api_key="gemini-secret-test", model="gemini-2.5-flash", session=session, max_retries=0)
    res = client.extract_json("system instructions", "paper chunks")
    assert res == {"facts": []}

    assert len(session.calls) == 1
    url, kwargs = session.calls[0][0][0], session.calls[0][1]
    assert url == f"{GEMINI_BASE_URL}/gemini-2.5-flash:generateContent"
    assert kwargs["headers"]["x-goog-api-key"] == "gemini-secret-test"
    assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert kwargs["json"]["generationConfig"]["temperature"] == 0.0
    assert kwargs["json"]["systemInstruction"]["parts"][0]["text"] == "system instructions"
    assert kwargs["json"]["contents"][0]["parts"][0]["text"] == "paper chunks"


def test_gemini_strips_markdown_code_fences():
    fence_wrapped = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "```json\n{\"facts\": [{\"id\": 1}]}\n```"}],
                    "role": "model",
                }
            }
        ]
    }
    session = FakeSession([FakeResponse(value=fence_wrapped)])
    client = GeminiClient(api_key="test-key", session=session, max_retries=0)
    result = client.extract_json("sys", "user")
    assert result == {"facts": [{"id": 1}]}


def test_gemini_token_usage_tracking():
    session = FakeSession([FakeResponse(value={
        "candidates": [{"content": {"parts": [{"text": "{\"facts\": []}"}]}}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 45},
    })])
    client = GeminiClient(api_key="test-key", session=session, max_retries=0)
    client.extract_json("sys", "user")
    assert client.observed_usage == {"prompt_tokens": 120, "completion_tokens": 45}


@pytest.mark.parametrize("status,error_body,expected_code", [
    (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}, "RATE_LIMITED"),
    (400, {"error": {"code": 400, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for quota metric"}}, "RATE_LIMITED"),
    (401, {"error": {"code": 401, "message": "Unauthorized"}}, "AUTH_ERROR"),
    (403, {"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "Permission denied"}}, "AUTH_ERROR"),
    (400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key."}}, "AUTH_ERROR"),
    (500, {"error": {"code": 500, "message": "Internal error"}}, "API_ERROR"),
])
def test_gemini_error_mapping_and_no_leak(status, error_body, expected_code):
    session = FakeSession([FakeResponse(status=status, value=error_body, text=json.dumps(error_body))])
    client = GeminiClient(api_key="SUPER-SECRET-GEMINI-TOKEN", session=session, max_retries=0)
    with pytest.raises(LLMError) as exc_info:
        client.extract_json("sys", "user")
    assert exc_info.value.code == expected_code
    assert "SUPER-SECRET-GEMINI-TOKEN" not in str(exc_info.value)


def test_gemini_transient_error_retry(monkeypatch):
    monkeypatch.setattr("researchgap.integrations.gemini_client.time.sleep", lambda *_: None)
    session = FakeSession([FakeResponse(status=503), FakeResponse()])
    client = GeminiClient(api_key="test-key", session=session, max_retries=1)
    res = client.extract_json("sys", "user")
    assert res == {"facts": []}
    assert len(session.calls) == 2


def test_gemini_missing_api_key():
    with pytest.raises(LLMError) as exc:
        GeminiClient(api_key="   ")
    assert exc.value.code == "MISSING_API_KEY"


def test_gemini_model_name_cleaning():
    session = FakeSession([FakeResponse()])
    client = GeminiClient(api_key="key", model="models/gemini-2.5-flash", session=session, max_retries=0)
    client.extract_json("sys", "user")
    assert session.calls[0][0][0] == f"{GEMINI_BASE_URL}/gemini-2.5-flash:generateContent"
