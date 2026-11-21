"""Google Gemini REST API adapter supporting structured JSON extraction.

Calls Google Generative Language v1beta REST API using standard HTTP requests.
Supports JSON mode, system instructions, token usage observation, and error mapping.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any

import requests

from researchgap.integrations.groq_client import JSONLanguageModel, LLMError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class GeminiClient:
    """Minimal Google Gemini REST API client implementing JSONLanguageModel."""

    api_key: str
    model: str = "gemini-2.5-flash"
    timeout_seconds: int = 60
    max_retries: int = 1
    session: requests.Session | None = None
    observed_usage: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise LLMError("MISSING_API_KEY", "Set GEMINI_API_KEYS before running analysis.")
        if not 5 <= self.timeout_seconds <= 180:
            raise ValueError("timeout_seconds must be between 5 and 180")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.session is None:
            self.session = requests.Session()

    def _clean_model_name(self) -> str:
        name = self.model.strip()
        if name.startswith("models/"):
            return name[7:]
        return name

    def extract_json(self, system_message: str, user_message: str) -> dict:
        clean_model = self._clean_model_name()
        url = f"{GEMINI_BASE_URL}/{clean_model}:generateContent"

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        if system_message.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_message}]
            }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        assert self.session is not None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
                    continue
                raise LLMError("NETWORK_ERROR", "Could not connect to the Gemini API.") from exc

            status = response.status_code

            # Transient server errors that can be retried on the same key
            if status in (500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 5))
                continue

            # Parse error body if status >= 400
            err_msg = ""
            err_status = ""
            if status >= 400:
                try:
                    err_json = response.json().get("error", {})
                    err_status = str(err_json.get("status", ""))
                    err_msg = str(err_json.get("message", ""))
                except Exception:
                    pass

            # Auth errors (HTTP 401, 403, or invalid API key status)
            if status in (401, 403) or "API_KEY_INVALID" in err_status or "api key not valid" in err_msg.lower():
                raise LLMError("AUTH_ERROR", f"Gemini API credentials were rejected: {err_msg or 'HTTP ' + str(status)}")

            # Rate limit or quota exhaustion (HTTP 429 or RESOURCE_EXHAUSTED)
            if status == 429 or "RESOURCE_EXHAUSTED" in err_status:
                combined_err = f"{err_status}: {err_msg}".lower()
                is_quota = (
                    "daily" in combined_err
                    or "per day" in combined_err
                    or "monthly" in combined_err
                    or "billing" in combined_err
                    or "requests_per_day" in combined_err
                    or "free_tier_requests_per_day" in combined_err
                    or "exceeded your current quota" in combined_err
                )
                if is_quota:
                    raise LLMError("QUOTA_EXHAUSTED", f"Gemini quota exhausted: {err_msg or 'HTTP 429 quota reached'}")

                # If temporary RPM/TPM limit and attempts remain, backoff and retry on same key
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.replace(".", "").isdigit() else (2**attempt + 0.5)
                    time.sleep(min(delay, 5))
                    continue

                raise LLMError("RATE_LIMITED", f"Gemini temporary rate limit reached: {err_msg or 'HTTP 429'}")

            if status >= 400:
                raise LLMError("API_ERROR", f"Gemini API returned HTTP {status}: {err_msg or 'Server error'}")

            # Successful response (HTTP 200)
            try:
                output = response.json()
                candidates = output.get("candidates", [])
                if not candidates:
                    prompt_feedback = output.get("promptFeedback", {})
                    block_reason = prompt_feedback.get("blockReason", "unknown")
                    raise LLMError("BAD_MODEL_RESPONSE", f"Gemini returned no candidates (blocked: {block_reason}).")

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise LLMError("BAD_MODEL_RESPONSE", "Gemini response did not contain text content.")

                text = parts[0]["text"].strip()
                # Remove markdown code fences if model returned them
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                decoded = json.loads(text)
                if not isinstance(decoded, dict):
                    raise TypeError("Gemini output is not a JSON object")

                # Record token usage in observed_usage for telemetry
                usage = output.get("usageMetadata", {})
                if isinstance(usage, dict):
                    in_count = usage.get("promptTokenCount")
                    out_count = usage.get("candidatesTokenCount")
                    if type(in_count) is int and in_count >= 0:
                        self.observed_usage["prompt_tokens"] = self.observed_usage.get("prompt_tokens", 0) + in_count
                    if type(out_count) is int and out_count >= 0:
                        self.observed_usage["completion_tokens"] = self.observed_usage.get("completion_tokens", 0) + out_count

                return decoded
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                if isinstance(exc, LLMError):
                    raise
                raise LLMError("BAD_MODEL_RESPONSE", "Gemini returned unusable JSON.") from exc

        raise LLMError("API_ERROR", "Gemini request failed after retries.")
