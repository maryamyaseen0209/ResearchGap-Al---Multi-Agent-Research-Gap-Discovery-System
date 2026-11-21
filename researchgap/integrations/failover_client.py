"""Multi-Key Failover Client: Gemini primary with automatic shift to Groq.

Supports multiple keys per provider (e.g. 3 Gemini keys + 3 Groq keys).
Order of execution:
1. Gemini Key 1 -> Gemini Key 2 -> Gemini Key 3 (rotates on rate limits / quota exceeded)
2. Automatically shifts to Groq Key 1 -> Groq Key 2 -> Groq Key 3 when Gemini limits are reached
3. Aggregates observed token usage across all keys for accurate telemetry
"""

from dataclasses import dataclass, field
import os
import time
from typing import Any

from dotenv import load_dotenv
import requests

from researchgap.integrations.gemini_client import GeminiClient
from researchgap.integrations.groq_client import GroqClient, JSONLanguageModel, LLMError


def mask_key(key: str) -> str:
    """Safely mask key for logs and UI status without leaking secrets."""
    clean = key.strip()
    if len(clean) <= 8:
        return "***"
    return f"{clean[:4]}...{clean[-4:]}"


@dataclass
class KeyTarget:
    """A configured provider key with metadata and health status."""
    provider: str  # "gemini" or "groq"
    key_index: int  # 1, 2, 3...
    client: JSONLanguageModel
    masked_key: str
    state: str = "ready"  # "ready", "rate_limited", "quota_exhausted", "auth_error"


def get_gemini_api_keys() -> list[str]:
    """Extract configured Gemini API keys in priority order.

    Supports:
    - Comma-separated: GEMINI_API_KEYS=key1,key2,key3
    - Numbered: GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    - Single: GEMINI_API_KEY
    """
    load_dotenv(override=False)
    keys: list[str] = []

    # 1. Comma-separated
    raw_list = os.getenv("GEMINI_API_KEYS", "").strip()
    if raw_list:
        for k in raw_list.split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    # If an explicit multi-key list was supplied, prioritize it
    if len(keys) >= 2:
        return keys

    # 2. Numbered keys (e.g. GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...)
    for i in range(1, 21):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)

    # 3. Single fallback if no explicit list/numbered keys provided
    if not keys:
        single = os.getenv("GEMINI_API_KEY", "").strip()
        if single and single not in keys:
            keys.append(single)

    return keys


def get_groq_api_keys() -> list[str]:
    """Extract configured Groq API keys in priority order.

    Supports:
    - Comma-separated: GROQ_API_KEYS=key1,key2,key3
    - Numbered: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
    - Single: GROQ_API_KEY
    """
    load_dotenv(override=False)
    keys: list[str] = []

    # 1. Comma-separated
    raw_list = os.getenv("GROQ_API_KEYS", "").strip()
    if raw_list:
        for k in raw_list.split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    # If an explicit multi-key list was supplied, prioritize it
    if len(keys) >= 2:
        return keys

    # 2. Numbered keys (e.g. GROQ_API_KEY_1, GROQ_API_KEY_2, ...)
    for i in range(1, 21):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)

    # 3. Single fallback if no explicit list/numbered keys provided
    if not keys:
        single = os.getenv("GROQ_API_KEY", "").strip()
        if single and single not in keys:
            keys.append(single)

    return keys


@dataclass
class FailoverLLMClient:
    """Seamlessly rotates through multiple keys and shifts between providers on rate limits.

    Implements JSONLanguageModel.
    """
    targets: list[KeyTarget]
    current_target_index: int = 0
    rotation_history: list[dict[str, Any]] = field(default_factory=list)
    observed_usage: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, float] = field(default_factory=dict)
    provider_error_msg: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.targets:
            raise LLMError("MISSING_API_KEY", "No valid Gemini or Groq API keys configured.")

    @property
    def model(self) -> str:
        """Returns the active client's model."""
        target = self.active_target
        return target.client.model if target else "none"

    @property
    def active_target(self) -> KeyTarget | None:
        if 0 <= self.current_target_index < len(self.targets):
            return self.targets[self.current_target_index]
        if self.targets:
            return self.targets[-1]
        return None

    @property
    def active_provider(self) -> str:
        target = self.active_target
        return target.provider if target else "none"

    @property
    def active_key_index(self) -> int:
        target = self.active_target
        return target.key_index if target else 0

    def is_provider_exhausted(self, provider: str) -> bool:
        """Check if all configured keys for a given provider are exhausted."""
        provider_targets = [t for t in self.targets if t.provider == provider]
        if not provider_targets:
            return True
        return all(t.state in ("quota_exhausted", "auth_error") for t in provider_targets)

    def reset_transient_limits(self) -> None:
        """Reset temporary rate limits and cooldowns while preserving permanent quota exhaustion."""
        self.cooldown_until.clear()
        for t in self.targets:
            if t.state == "rate_limited":
                t.state = "ready"
        for idx, t in enumerate(self.targets):
            if t.state not in ("quota_exhausted", "auth_error"):
                self.current_target_index = idx
                break

    def get_telemetry(self) -> dict[str, Any]:
        """Diagnostic state summary for UI progress and logs."""
        target = self.active_target
        providers = sorted({t.provider for t in self.targets})
        now = time.time()
        return {
            "active_provider": target.provider if target else "none",
            "active_key_index": target.key_index if target else 0,
            "active_key_masked": target.masked_key if target else "none",
            "active_model": self.model,
            "provider_status": {
                p: "exhausted" if self.is_provider_exhausted(p)
                else ("cooldown" if self.cooldown_until.get(p, 0) > now else "active")
                for p in providers
            },
            "cooldown_remaining": {
                p: max(0, int(self.cooldown_until[p] - now))
                for p in providers if self.cooldown_until.get(p, 0) > now
            },
            "observed_usage": dict(self.observed_usage),
            "rotations_count": len(self.rotation_history),
            "keys_summary": [
                {
                    "provider": t.provider,
                    "key_index": t.key_index,
                    "state": t.state,
                    "masked_key": t.masked_key,
                }
                for t in self.targets
            ],
        }

    def status_summary(self) -> str:
        """User-friendly status for UI display."""
        target = self.active_target
        if not target:
            return "No active LLM keys available"

        gemini_targets = [t for t in self.targets if t.provider == "gemini"]
        groq_targets = [t for t in self.targets if t.provider == "groq"]
        now = time.time()

        parts = []
        if target.provider == "gemini":
            parts.append(f"Active: Gemini (Key {target.key_index} of {len(gemini_targets)})")
            if groq_targets:
                groq_ready = sum(1 for t in groq_targets if t.state != "quota_exhausted")
                parts.append(f"Failover: Groq ({groq_ready} key{'s' if groq_ready > 1 else ''} ready)")
        else:
            parts.append(f"Active: Groq (Key {target.key_index} of {len(groq_targets)})")
            if gemini_targets:
                parts.append(f"Shifted from Gemini ({len(gemini_targets)} key{'s' if len(gemini_targets) > 1 else ''} exhausted)")

        # Add cooldown notices if active
        for p, cd in self.cooldown_until.items():
            if cd > now:
                parts.append(f"[{p.capitalize()} cooling down: {int(cd - now)}s]")

        return " | ".join(parts)

    def is_provider_rate_limited(self, provider: str) -> bool:
        """Check if all configured keys for a given provider have hit limits."""
        provider_targets = [t for t in self.targets if t.provider == provider]
        if not provider_targets:
            return True
        return all(t.state in ("rate_limited", "quota_exhausted", "auth_error") for t in provider_targets)

    def extract_json(self, system_message: str, user_message: str) -> dict:
        """Extract JSON with automatic key rotation, bounded retry, and provider shift."""
        while self.current_target_index < len(self.targets):
            target = self.targets[self.current_target_index]
            now = time.time()

            # Skip targets already known to be exhausted
            if target.state in ("quota_exhausted", "auth_error"):
                self.current_target_index += 1
                continue

            # If all keys for this provider have been exhausted or rate-limited and cooldown is active
            if self.is_provider_rate_limited(target.provider) and self.cooldown_until.get(target.provider, 0) > now:
                alt_idx = None
                for idx, t in enumerate(self.targets):
                    if t.state not in ("quota_exhausted", "auth_error") and (
                        not self.is_provider_rate_limited(t.provider)
                        or self.cooldown_until.get(t.provider, 0) <= now
                    ):
                        alt_idx = idx
                        break
                if alt_idx is not None:
                    self.current_target_index = alt_idx
                    target = self.targets[self.current_target_index]
                else:
                    # All providers in cooldown or exhausted
                    cds = [self.cooldown_until[p] - now for p in self.cooldown_until if self.cooldown_until[p] > now]
                    min_wait = max(1.0, min(cds)) if cds else 10.0
                    raise LLMError(
                        "RATE_LIMITED",
                        f"All configured LLM providers are currently cooling down ({int(min_wait)}s remaining). "
                        "Completed batches are saved in cache. Resume once cooldown expires.",
                    )

            client = target.client
            before = getattr(client, "observed_usage", {})
            before_in = before.get("prompt_tokens", 0)
            before_out = before.get("completion_tokens", 0)

            try:
                result = client.extract_json(system_message, user_message)

                target.state = "ready"
                after = getattr(client, "observed_usage", {})
                delta_in = after.get("prompt_tokens", 0) - before_in
                delta_out = after.get("completion_tokens", 0) - before_out
                if delta_in > 0:
                    self.observed_usage["prompt_tokens"] = self.observed_usage.get("prompt_tokens", 0) + delta_in
                if delta_out > 0:
                    self.observed_usage["completion_tokens"] = self.observed_usage.get("completion_tokens", 0) + delta_out

                return result

            except LLMError as exc:
                old_target = target

                if exc.code == "QUOTA_EXHAUSTED":
                    old_target.state = "quota_exhausted"
                    self.provider_error_msg[old_target.provider] = str(exc)
                elif exc.code == "AUTH_ERROR":
                    old_target.state = "auth_error"
                    self.provider_error_msg[old_target.provider] = str(exc)
                elif exc.code == "RATE_LIMITED":
                    old_target.state = "rate_limited"
                    delay = getattr(exc, "retry_after", None) or 10.0
                    if self.is_provider_rate_limited(old_target.provider):
                        self.cooldown_until[old_target.provider] = time.time() + float(delay)
                    self.provider_error_msg[old_target.provider] = str(exc)
                else:
                    old_target.state = "error"

                # If current provider is now exhausted of all keys, jump directly to next provider
                if self.is_provider_exhausted(old_target.provider):
                    next_idx = None
                    for idx in range(self.current_target_index + 1, len(self.targets)):
                        if self.targets[idx].provider != old_target.provider and self.targets[idx].state not in ("quota_exhausted", "auth_error"):
                            next_idx = idx
                            break
                    if next_idx is not None:
                        self.current_target_index = next_idx
                    else:
                        self.current_target_index += 1
                else:
                    self.current_target_index += 1

                if self.current_target_index < len(self.targets):
                    new_target = self.targets[self.current_target_index]
                    self.rotation_history.append({
                        "timestamp": time.time(),
                        "from_provider": old_target.provider,
                        "from_key": old_target.key_index,
                        "to_provider": new_target.provider,
                        "to_key": new_target.key_index,
                        "reason": exc.code,
                        "error_message": str(exc),
                    })
                    continue
                else:
                    all_quota = all(t.state in ("quota_exhausted", "auth_error") for t in self.targets)
                    if all_quota:
                        raise LLMError(
                            "QUOTA_EXHAUSTED",
                            "All configured Gemini and Groq API keys reached daily or monthly quota limits."
                        ) from exc
                    else:
                        raise LLMError(
                            "RATE_LIMITED",
                            "All configured Gemini and Groq API keys reached rate limits or quota. "
                            "Completed batches have been preserved and can be resumed."
                        ) from exc

        all_quota = all(t.state in ("quota_exhausted", "auth_error") for t in self.targets)
        err_code = "QUOTA_EXHAUSTED" if all_quota else "RATE_LIMITED"
        raise LLMError(err_code, "All configured Gemini and Groq API keys are currently unavailable.")


def create_llm_client(
    gemini_keys: list[str] | None = None,
    groq_keys: list[str] | None = None,
    gemini_model: str | None = None,
    groq_model: str | None = None,
    session: requests.Session | None = None,
) -> FailoverLLMClient:
    """Create a unified failover client prioritizing Gemini keys and failing over to Groq."""
    load_dotenv(override=False)

    if gemini_keys is None:
        gemini_keys = get_gemini_api_keys()
    if groq_keys is None:
        groq_keys = get_groq_api_keys()

    if not gemini_keys and not groq_keys:
        raise LLMError(
            "MISSING_API_KEY",
            "Set GEMINI_API_KEYS (or GEMINI_API_KEY_1..3) or GROQ_API_KEYS in .env before running analysis."
        )

    g_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    q_model = groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    targets: list[KeyTarget] = []

    # 1. Primary: Gemini keys (Key 1, Key 2, Key 3...)
    for idx, key in enumerate(gemini_keys, start=1):
        targets.append(
            KeyTarget(
                provider="gemini",
                key_index=idx,
                client=GeminiClient(api_key=key, model=g_model, session=session),
                masked_key=mask_key(key),
            )
        )

    # 2. Secondary / Failover: Groq keys (Key 1, Key 2, Key 3...)
    for idx, key in enumerate(groq_keys, start=1):
        targets.append(
            KeyTarget(
                provider="groq",
                key_index=idx,
                client=GroqClient(api_key=key, model=q_model, session=session),
                masked_key=mask_key(key),
            )
        )

    return FailoverLLMClient(targets=targets)
