"""Observable LLM calls with measured latency and optional token-based pricing.

No invented token counts or dollar amounts: if provider usage is missing, cost
is None. A pricing estimate requires explicit user-supplied rates.
"""

from dataclasses import dataclass
import json
from time import perf_counter

from researchgap.integrations.groq_client import JSONLanguageModel


@dataclass
class MeteredLLM:
    delegate: JSONLanguageModel
    calls: int = 0
    errors: int = 0
    input_characters: int = 0
    output_characters: int = 0
    seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def model(self):
        return self.delegate.model

    def extract_json(self, system_message: str, user_message: str) -> dict:
        self.calls += 1
        self.input_characters += len(system_message) + len(user_message)
        begin = perf_counter()
        before = getattr(self.delegate, "observed_usage", None)
        before_in = before.get("prompt_tokens", 0) if isinstance(before, dict) else 0
        before_out = before.get("completion_tokens", 0) if isinstance(before, dict) else 0
        try:
            result = self.delegate.extract_json(system_message, user_message)
            self.output_characters += len(json.dumps(result, ensure_ascii=False))
            return result
        except Exception:
            self.errors += 1
            raise
        finally:
            self.seconds += perf_counter() - begin
            after = getattr(self.delegate, "observed_usage", None)
            if isinstance(after, dict) and "prompt_tokens" in after and "completion_tokens" in after:
                a = after["prompt_tokens"] - before_in
                b = after["completion_tokens"] - before_out
                if a >= 0 and b >= 0:
                    self.input_tokens = (self.input_tokens or 0) + a
                    self.output_tokens = (self.output_tokens or 0) + b

    def measurements(self, *, input_usd_per_million: float | None = None,
                     output_usd_per_million: float | None = None) -> dict:
        cost = None
        if (self.input_tokens is not None and self.output_tokens is not None
                and input_usd_per_million is not None and output_usd_per_million is not None
                and input_usd_per_million >= 0 and output_usd_per_million >= 0):
            cost = (self.input_tokens * input_usd_per_million
                    + self.output_tokens * output_usd_per_million) / 1_000_000
        return {
            "model": self.model, "calls": self.calls, "failed_calls": self.errors,
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "llm_latency_seconds": round(self.seconds, 4),
            "observed_input_tokens": self.input_tokens,
            "observed_output_tokens": self.output_tokens,
            "estimated_cost_usd": (round(cost, 8) if cost is not None else None),
            "notes": "Cost requires API-reported token counts and explicit model pricing; excludes OpenAlex and other fees.",
        }


@dataclass
class MeteredSearch:
    """Counts OpenAlex search operations without logging private query text."""
    delegate: object
    calls: int = 0
    errors: int = 0
    seconds: float = 0.0

    def search(self, query: str, limit: int = 5):
        self.calls += 1
        began = perf_counter()
        try:
            return self.delegate.search(query, limit=limit)
        except Exception:
            self.errors += 1
            raise
        finally:
            self.seconds += perf_counter() - began

    def measurements(self):
        return {"search_calls": self.calls, "search_errors": self.errors,
                "search_latency_seconds": round(self.seconds, 4)}
