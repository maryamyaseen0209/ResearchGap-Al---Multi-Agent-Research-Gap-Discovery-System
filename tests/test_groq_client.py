"""Mock HTTP tests: verify Groq payload, JSON parsing, failure modes, retries."""
import json

import pytest
import requests

from researchgap.integrations.groq_client import GroqClient, LLMError, GROQ_ENDPOINT


class FakeResponse:
    def __init__(self, status=200, value=None, headers=None):
        self.status_code = status
        self.value = value if value is not None else {
            "choices": [{"message": {"content": json.dumps({"facts": []})}}]
        }
        self.headers = headers or {}

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


def test_valid_groq_request_and_json():
    session = FakeSession([FakeResponse()])
    client = GroqClient(api_key="only-for-tests", session=session, max_retries=0)
    assert client.extract_json("json please", "paper") == {"facts": []}
    args, kwargs = session.calls[0]
    assert args[0] == GROQ_ENDPOINT
    assert kwargs["json"]["response_format"] == {"type": "json_object"}
    assert kwargs["json"]["model"] == "openai/gpt-oss-120b"
    assert kwargs["json"]["reasoning_effort"] == "low"
    assert "reasoning_format" not in kwargs["json"]
    assert kwargs["json"]["temperature"] == 0
    assert kwargs["headers"]["Authorization"] == "Bearer only-for-tests"
    assert kwargs["timeout"] == 60


@pytest.mark.parametrize("status,expected", [(401,"AUTH_ERROR"),(403,"AUTH_ERROR"),
                                              (429,"RATE_LIMITED"),(500,"API_ERROR"),(400,"API_ERROR")])
def test_http_errors_do_not_leak_token(status, expected):
    session = FakeSession([FakeResponse(status=status)])
    client = GroqClient(api_key="SECRET-DONT-LEAK", session=session, max_retries=0)
    with pytest.raises(LLMError) as ex:
        client.extract_json("prompt", "source")
    assert ex.value.code == expected
    assert "SECRET-DONT-LEAK" not in str(ex.value)


def test_retry_on_transient_error(monkeypatch):
    monkeypatch.setattr("researchgap.integrations.groq_client.time.sleep", lambda *_: None)
    session = FakeSession([FakeResponse(status=503), FakeResponse()])
    client = GroqClient(api_key="test", session=session, max_retries=1)
    assert client.extract_json("system", "user") == {"facts": []}
    assert len(session.calls) == 2


def test_network_error_is_retried_and_reported(monkeypatch):
    monkeypatch.setattr("researchgap.integrations.groq_client.time.sleep", lambda *_: None)
    session = FakeSession([requests.Timeout("network"), requests.Timeout("network")])
    client = GroqClient(api_key="test", session=session, max_retries=1)
    with pytest.raises(LLMError) as ex:
        client.extract_json("system", "user")
    assert ex.value.code == "NETWORK_ERROR"
    assert len(session.calls) == 2


@pytest.mark.parametrize("payload", [
    {"choices": []},
    {"choices": [{"message": {"content": "not a JSON payload"}}]},
    {"choices": [{"message": {"content": "[]"}}]},
])
def test_bad_model_response(payload):
    session = FakeSession([FakeResponse(value=payload)])
    client = GroqClient(api_key="test", session=session, max_retries=0)
    with pytest.raises(LLMError) as ex:
        client.extract_json("system", "user")
    assert ex.value.code == "BAD_MODEL_RESPONSE"


def test_no_api_key_rejected_without_request():
    with pytest.raises(LLMError) as ex:
        GroqClient(api_key="  ")
    assert ex.value.code == "MISSING_API_KEY"


def test_invalid_retry_settings():
    with pytest.raises(ValueError):
        GroqClient(api_key="test", max_retries=20)
    with pytest.raises(ValueError):
        GroqClient(api_key="test", timeout_seconds=1)


def test_other_models_do_not_receive_unsupported_reasoning_effort():
    session = FakeSession([FakeResponse()])
    client = GroqClient(api_key="test", model="custom-model", session=session, max_retries=0)
    client.extract_json("system", "user")
    assert "reasoning_effort" not in session.calls[0][1]["json"]


def test_actual_provider_usage_is_recorded_without_leaking_api_key():
    session = FakeSession([FakeResponse(value={
        "choices": [{"message": {"content": '{"facts": []}'}}],
        "usage": {"prompt_tokens": 72, "completion_tokens": 34},
    })])
    client = GroqClient(api_key="PRIVATE_TEST_KEY", session=session)
    assert client.extract_json("system", "user") == {"facts": []}
    assert client.observed_usage == {"prompt_tokens": 72, "completion_tokens": 34}
