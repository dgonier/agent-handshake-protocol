"""OpenRouter chat-model helper — one of three "natural model sources" for AHP.

OpenRouter (https://openrouter.ai) is OpenAI-API-compatible, so any
LangChain integration that speaks OpenAI works against it by changing
``base_url``. This module's only job is the boilerplate: read the API
key from env, pin the base URL, and hand back a ``ChatOpenAI``
instance that ``ReactAgent.from_profile`` / ``DeepAgent.from_profile``
can consume unchanged.

Configuration via environment / ``.env``:

* ``OPENROUTER_API_KEY`` — required.
* ``OPENROUTER_MODEL`` — optional default model id (falls back to a
  reasonable default if unset).
* ``OPENROUTER_BASE_URL`` — usually leave alone; override only if
  you're proxying through your own gateway.

Pick the right source for the job:

* **Bedrock** when you want hosted Claude on AWS infra and have AWS
  credentials handy (``ahp/llm/bedrock.py``).
* **OpenRouter** when you want to A/B test across many model families
  without re-wiring (this module).
* **Modal** when you want to run your own GPUs — base model + LoRAs
  + custom inference (host the agent on Modal, expose as an AHP node;
  see ``ahp.adapters.deep_agent`` and ``ahp.llm.recipe`` for the
  composition layer).
"""

from __future__ import annotations

import os
from functools import lru_cache


OPENROUTER_DEFAULT_MODEL_ID: str = "anthropic/claude-3.5-sonnet"
"""Default model id when ``OPENROUTER_MODEL`` is unset."""

OPENROUTER_DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"


def _load_dotenv_quiet() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    load_dotenv(override=False)


def default_openrouter_model_id() -> str:
    _load_dotenv_quiet()
    return os.environ.get("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL_ID)


def default_base_url() -> str:
    _load_dotenv_quiet()
    return os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)


def has_openrouter_credentials() -> bool:
    """Cheap pre-flight: is ``OPENROUTER_API_KEY`` set?"""
    _load_dotenv_quiet()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


@lru_cache(maxsize=16)
def openrouter_chat_model(
    model: str | None = None,
    base_url: str | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
):
    """Construct a cached ``ChatOpenAI`` pointed at OpenRouter.

    Returns a LangChain-compatible chat model that
    :meth:`ReactAgent.from_profile` and :meth:`DeepAgent.from_profile`
    consume directly. Cached per model id + base URL so repeated calls
    in one process reuse the underlying HTTP client.
    """
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    _load_dotenv_quiet()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Sign in at https://openrouter.ai, "
            "create a key, export it, and try again."
        )

    return ChatOpenAI(
        model=model or default_openrouter_model_id(),
        base_url=base_url or default_base_url(),
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
