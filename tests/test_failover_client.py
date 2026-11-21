"""Unit tests for Multi-Key Failover Client (Gemini 1..3 -> Groq 1..3)."""
import pytest

from researchgap.evaluation.telemetry import MeteredLLM
from researchgap.integrations.failover_client import (
    FailoverLLMClient,
    KeyTarget,
    create_llm_client,
    get_gemini_api_keys,
    get_groq_api_keys,
    mask_key,
)
from researchgap.integrations.groq_client import LLMError


class DummyClient:
    def __init__(self, model: str, responses: list):
        self.model = model
        self.responses = list(responses)
        self.observed_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.calls = 0

    def extract_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if not self.responses:
            raise LLMError("RATE_LIMITED", f"{self.model} out of responses")
        res = self.responses.pop(0)
        if isinstance(res, Exception):
            raise res
        self.observed_usage["prompt_tokens"] += 50
        self.observed_usage["completion_tokens"] += 25
        return res


def test_mask_key_security():
    assert mask_key("AIzaSy1234567890abcdef") == "AIza...cdef"
    assert mask_key("gsk_1234567890abcdef") == "gsk_...cdef"
    assert mask_key("short") == "***"


def test_key_parsing_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "gemini_a, gemini_b, gemini_c")
    monkeypatch.setenv("GROQ_API_KEYS", "groq_a, groq_b, groq_c")

    g_keys = get_gemini_api_keys()
    q_keys = get_groq_api_keys()

    assert g_keys == ["gemini_a", "gemini_b", "gemini_c"]
    assert q_keys == ["groq_a", "groq_b", "groq_c"]


def test_numbered_key_parsing_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.setenv("GROQ_API_KEYS", "")
    monkeypatch.setenv("GEMINI_API_KEY_1", "gem1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gem2")
    monkeypatch.setenv("GROQ_API_KEY_1", "grq1")

    assert get_gemini_api_keys()[:2] == ["gem1", "gem2"]
    assert get_groq_api_keys()[0] == "grq1"


def test_gemini_key_rotation_on_rate_limit():
    """Gemini Key 1 hits rate limit -> automatically rotates to Gemini Key 2."""
    g1 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "429 Quota Exceeded")])
    g2 = DummyClient("gemini-2.5-flash", [{"facts": ["from_gemini_2"]}])
    g3 = DummyClient("gemini-2.5-flash", [{"facts": ["from_gemini_3"]}])

    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("gemini", 2, g2, "gem2"),
        KeyTarget("gemini", 3, g3, "gem3"),
    ]
    failover = FailoverLLMClient(targets=targets)

    assert failover.active_provider == "gemini"
    assert failover.active_key_index == 1

    result = failover.extract_json("sys", "user")

    assert result == {"facts": ["from_gemini_2"]}
    assert failover.active_key_index == 2
    assert len(failover.rotation_history) == 1
    assert failover.rotation_history[0]["from_key"] == 1
    assert failover.rotation_history[0]["to_key"] == 2
    assert failover.rotation_history[0]["reason"] == "RATE_LIMITED"


def test_automatic_shift_from_gemini_to_groq():
    """All 3 Gemini keys hit rate limits -> automatically shifts to Groq Key 1."""
    g1 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "Gemini 1 quota")])
    g2 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "Gemini 2 quota")])
    g3 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "Gemini 3 quota")])

    q1 = DummyClient("openai/gpt-oss-120b", [{"facts": ["from_groq_1"]}])
    q2 = DummyClient("openai/gpt-oss-120b", [{"facts": ["from_groq_2"]}])
    q3 = DummyClient("openai/gpt-oss-120b", [{"facts": ["from_groq_3"]}])

    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("gemini", 2, g2, "gem2"),
        KeyTarget("gemini", 3, g3, "gem3"),
        KeyTarget("groq", 1, q1, "groq1"),
        KeyTarget("groq", 2, q2, "groq2"),
        KeyTarget("groq", 3, q3, "groq3"),
    ]
    failover = FailoverLLMClient(targets=targets)

    result = failover.extract_json("sys", "user")

    assert result == {"facts": ["from_groq_1"]}
    assert failover.active_provider == "groq"
    assert failover.active_key_index == 1
    assert failover.model == "openai/gpt-oss-120b"
    assert len(failover.rotation_history) == 3
    # Check that third rotation was from Gemini to Groq
    shift_event = failover.rotation_history[2]
    assert shift_event["from_provider"] == "gemini"
    assert shift_event["to_provider"] == "groq"
    assert shift_event["to_key"] == 1


def test_groq_internal_rotation_when_shifted():
    """Once shifted to Groq, if Groq Key 1 hits limit, rotates to Groq Key 2."""
    g1 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "Gemini 1 quota")])
    q1 = DummyClient("openai/gpt-oss-120b", [LLMError("RATE_LIMITED", "Groq 1 429")])
    q2 = DummyClient("openai/gpt-oss-120b", [{"facts": ["from_groq_2"]}])

    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("groq", 1, q1, "groq1"),
        KeyTarget("groq", 2, q2, "groq2"),
    ]
    failover = FailoverLLMClient(targets=targets)
    result = failover.extract_json("sys", "user")

    assert result == {"facts": ["from_groq_2"]}
    assert failover.active_provider == "groq"
    assert failover.active_key_index == 2


def test_total_exhaustion_raises_rate_limited():
    """When all Gemini and all Groq keys are exhausted, clear error is raised."""
    g1 = DummyClient("gemini-2.5-flash", [LLMError("RATE_LIMITED", "G1")])
    q1 = DummyClient("openai/gpt-oss-120b", [LLMError("RATE_LIMITED", "Q1")])

    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("groq", 1, q1, "groq1"),
    ]
    failover = FailoverLLMClient(targets=targets)

    with pytest.raises(LLMError) as exc_info:
        failover.extract_json("sys", "user")
    assert exc_info.value.code == "RATE_LIMITED"
    assert "All configured Gemini and Groq API keys reached rate limits" in str(exc_info.value)


def test_status_summary_display():
    g1 = DummyClient("gemini-2.5-flash", [])
    g2 = DummyClient("gemini-2.5-flash", [])
    g3 = DummyClient("gemini-2.5-flash", [])
    q1 = DummyClient("openai/gpt-oss-120b", [])
    q2 = DummyClient("openai/gpt-oss-120b", [])
    q3 = DummyClient("openai/gpt-oss-120b", [])

    targets = [
        KeyTarget("gemini", 1, g1, "gem1"),
        KeyTarget("gemini", 2, g2, "gem2"),
        KeyTarget("gemini", 3, g3, "gem3"),
        KeyTarget("groq", 1, q1, "groq1"),
        KeyTarget("groq", 2, q2, "groq2"),
        KeyTarget("groq", 3, q3, "groq3"),
    ]
    failover = FailoverLLMClient(targets=targets)

    # Initial state: Gemini Key 1 active, 3 Groq failover keys ready
    status = failover.status_summary()
    assert "Active: Gemini (Key 1 of 3)" in status
    assert "Failover: Groq (3 keys ready)" in status

    # Simulate shift to Groq Key 2
    failover.current_target_index = 4
    status_shifted = failover.status_summary()
    assert "Active: Groq (Key 2 of 3)" in status_shifted
    assert "Shifted from Gemini" in status_shifted


def test_telemetry_metered_llm_integration():
    """Verifies that FailoverLLMClient seamlessly integrates with MeteredLLM."""
    g1 = DummyClient("gemini-2.5-flash", [{"facts": ["ok"]}])
    targets = [KeyTarget("gemini", 1, g1, "gem1")]
    failover = FailoverLLMClient(targets=targets)

    metered = MeteredLLM(failover)
    res = metered.extract_json("system", "user")
    assert res == {"facts": ["ok"]}

    measurements = metered.measurements()
    assert measurements["calls"] == 1
    assert measurements["observed_input_tokens"] == 50
    assert measurements["observed_output_tokens"] == 25
    assert measurements["model"] == "gemini-2.5-flash"
