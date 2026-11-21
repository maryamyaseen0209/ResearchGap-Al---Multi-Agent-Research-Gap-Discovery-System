"""Integrations with external LLM providers and scholarly databases."""

from researchgap.integrations.gemini_client import GeminiClient
from researchgap.integrations.groq_client import GroqClient, JSONLanguageModel, LLMError, RealGroqClient
from researchgap.integrations.failover_client import (
    FailoverLLMClient,
    KeyTarget,
    create_llm_client,
    get_gemini_api_keys,
    get_groq_api_keys,
)

__all__ = [
    "GeminiClient",
    "GroqClient",
    "RealGroqClient",
    "FailoverLLMClient",
    "KeyTarget",
    "create_llm_client",
    "get_gemini_api_keys",
    "get_groq_api_keys",
    "JSONLanguageModel",
    "LLMError",
]
