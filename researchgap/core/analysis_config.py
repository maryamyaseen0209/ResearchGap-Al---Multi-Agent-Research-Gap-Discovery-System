"""Bounded, validated analysis limits; no unannounced paper truncation."""

import os
from pydantic import BaseModel, Field, model_validator


class AnalysisSettings(BaseModel):
    max_chunks_per_paper: int = Field(default=60, ge=1, le=150)
    max_characters_per_paper: int = Field(default=120_000, ge=200, le=250_000)
    max_characters_per_request: int = Field(default=12_000, ge=200, le=40_000)
    max_facts_per_category: int = Field(default=15, ge=1, le=50)

    @model_validator(mode="after")
    def validate_limits(self) -> "AnalysisSettings":
        if self.max_characters_per_request > self.max_characters_per_paper:
            raise ValueError("max_characters_per_request must not exceed max_characters_per_paper")
        return self


def configured_api_key() -> str | None:
    """Returns the first configured active API key (Gemini primary, Groq secondary)."""
    from dotenv import load_dotenv

    load_dotenv(override=False)
    # Check Gemini keys first (primary provider)
    gemini_keys = os.getenv("GEMINI_API_KEYS", "").strip()
    if gemini_keys:
        first = gemini_keys.split(",")[0].strip()
        if first:
            return first
    for i in range(1, 21):
        g = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if g:
            return g
    g_single = os.getenv("GEMINI_API_KEY", "").strip()
    if g_single:
        return g_single

    # Check Groq keys (failover provider)
    groq_keys = os.getenv("GROQ_API_KEYS", "").strip()
    if groq_keys:
        first = groq_keys.split(",")[0].strip()
        if first:
            return first
    for i in range(1, 21):
        q = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if q:
            return q
    return os.getenv("GROQ_API_KEY", "").strip() or None


def get_llm_configuration_summary() -> dict:
    """Returns metadata about configured Gemini and Groq API keys."""
    from researchgap.integrations.failover_client import get_gemini_api_keys, get_groq_api_keys

    gemini_keys = get_gemini_api_keys()
    groq_keys = get_groq_api_keys()
    g_count = len(gemini_keys)
    q_count = len(groq_keys)

    if g_count > 0 and q_count > 0:
        desc = f"Gemini ({g_count} key{'s' if g_count > 1 else ''}) → Groq failover ({q_count} key{'s' if q_count > 1 else ''})"
        primary = "gemini"
    elif g_count > 0:
        desc = f"Gemini ({g_count} key{'s' if g_count > 1 else ''})"
        primary = "gemini"
    elif q_count > 0:
        desc = f"Groq ({q_count} key{'s' if q_count > 1 else ''})"
        primary = "groq"
    else:
        desc = "No API keys configured"
        primary = None

    return {
        "gemini_keys_count": g_count,
        "groq_keys_count": q_count,
        "has_keys": (g_count + q_count) > 0,
        "primary_provider": primary,
        "summary": desc,
    }

