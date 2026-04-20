"""
Centralized LLM provider factory.

Switches between Gemini (default) and Ollama based on the LLM_PROVIDER
environment variable. All example scripts import get_llm() from here
instead of instantiating the LLM directly.

Usage in .env:
    # Default: Gemini (requires GOOGLE_API_KEY)
    LLM_PROVIDER=gemini

    # Ollama (requires ollama running locally)
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3.1:8b
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_llm(temperature: float = 0, **kwargs) -> BaseChatModel:
    """Returns a chat LLM instance based on the LLM_PROVIDER env var.

    Args:
        temperature: Sampling temperature (0 = deterministic).
        **kwargs: Extra arguments passed to the underlying model constructor
                  (e.g. max_output_tokens for Gemini).

    Supported providers:
        gemini  - Google Gemini via langchain-google-genai (default).
        ollama  - Local models via langchain-ollama.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    # Default: Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)
