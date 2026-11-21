"""Production-oriented minimal Groq chat-completions adapter.

The real Groq API is called only when the user supplies a GROQ_API_KEY.
The tests use a fake HTTP session and never require a paid API account.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Protocol

import requests

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(Exception):
    """Safe, non-secret diagnostic error for a model request."""

    def __init__(
        self,
        code: str,
        message: str,
        retry_after: float | None = None,
        provider: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.provider = provider


class JSONLanguageModel(Protocol):
    model: str

    def extract_json(self, system_message: str, user_message: str) -> dict:
        ...


@dataclass
class GroqClient:
    api_key: str
    model: str = "openai/gpt-oss-120b"
    timeout_seconds: int = 60
    max_retries: int = 2
    session: requests.Session | None = None
    observed_usage: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise LLMError("MISSING_API_KEY", "Set GROQ_API_KEY before running paper analysis.")
        if not 5 <= self.timeout_seconds <= 180:
            raise ValueError("timeout_seconds must be between 5 and 180")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.session is None:
            self.session = requests.Session()

    def extract_json(self, system_message: str, user_message: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }
        # GPT-OSS is a reasoning model; use a small supported reasoning budget.
        # It does NOT accept the 'reasoning_format' parameter on Groq.
        if self.model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            payload["reasoning_effort"] = "low"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        assert self.session is not None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    GROQ_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
                    continue
                raise LLMError("NETWORK_ERROR", "Could not connect to the LLM API.") from exc

            status = response.status_code
            if status in (500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 5))
                continue

            err_msg = ""
            if status >= 400:
                try:
                    err_data = response.json().get("error", {})
                    err_msg = str(err_data.get("message", "") if isinstance(err_data, dict) else err_data)
                except Exception:
                    pass

            if status in (401, 403):
                raise LLMError("AUTH_ERROR", f"Groq API credentials were rejected: {err_msg or 'HTTP ' + str(status)}")

            if status == 429:
                err_lower = err_msg.lower()
                is_quota = (
                    "quota" in err_lower
                    or "insufficient_quota" in err_lower
                    or "daily" in err_lower
                    or "exceeded your current quota" in err_lower
                )
                if is_quota:
                    raise LLMError("QUOTA_EXHAUSTED", f"Groq quota exhausted: {err_msg or 'HTTP 429 quota reached'}")

                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.replace(".", "").isdigit() else (2**attempt + 0.5)
                    time.sleep(min(delay, 5))
                    continue

                raise LLMError("RATE_LIMITED", f"Groq rate limit reached: {err_msg or 'HTTP 429'}")

            if status >= 400:
                raise LLMError("API_ERROR", f"The LLM API returned HTTP {status}: {err_msg or 'Server error'}")

            try:
                output = response.json()
                text = output["choices"][0]["message"]["content"]
                decoded = json.loads(text)
                if not isinstance(decoded, dict):
                    raise TypeError("LLM output is not an object")
                usage = output.get("usage", {})
                if isinstance(usage, dict):
                    for source_key, target_key in (("prompt_tokens", "prompt_tokens"),
                                                  ("completion_tokens", "completion_tokens")):
                        count = usage.get(source_key)
                        if type(count) is int and count >= 0:
                            self.observed_usage[target_key] = self.observed_usage.get(target_key, 0) + count
                return decoded
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                raise LLMError("BAD_MODEL_RESPONSE", "The LLM returned unusable JSON.") from exc

        raise LLMError("API_ERROR", "LLM request failed after retries.")


RealGroqClient = GroqClient

