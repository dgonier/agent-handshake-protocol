"""AWS Bedrock chat-model helper.

Reads configuration from environment variables (after a best-effort
``.env`` load):

* ``AWS_REGION`` — required-ish; falls back to ``us-east-1``.
* ``BEDROCK_MODEL_ID`` — Bedrock model identifier; falls back to
  :data:`BEDROCK_DEFAULT_MODEL_ID`.
* ``AWS_PROFILE`` — optional named profile from ``aws configure``.
  (Credentials themselves are pulled by ``boto3`` from the usual chain;
  this module never touches keys directly.)
"""

from __future__ import annotations

import os
from functools import lru_cache


BEDROCK_DEFAULT_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
"""Default Bedrock model id when none is supplied or set via env."""


def _load_dotenv_quiet() -> None:
    """Load a *project-local* ``.env`` if one sits next to CWD.

    Earlier versions walked all the way up to ``$HOME``; that turned out
    to be a footgun on developer machines where ``~/.env`` carries
    unrelated, sometimes malformed, AWS keys that fight with the boto3
    credentials chain. We now stop at CWD — explicit project envs only.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    cwd_env = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(cwd_env):
        load_dotenv(dotenv_path=cwd_env, override=False)


def default_bedrock_model_id() -> str:
    _load_dotenv_quiet()
    return os.environ.get("BEDROCK_MODEL_ID", BEDROCK_DEFAULT_MODEL_ID)


def default_region() -> str:
    _load_dotenv_quiet()
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def has_aws_credentials() -> bool:
    """Cheap pre-flight check: does boto3 see *any* usable credentials?

    Returns False if boto3 isn't installed, no credentials are discoverable,
    or the discovery itself raises. Useful for tests / demos that want to
    skip gracefully when running outside AWS.
    """
    _load_dotenv_quiet()
    try:
        import boto3  # type: ignore[import-not-found]
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
        return session.get_credentials() is not None
    except Exception:
        return False


@lru_cache(maxsize=8)
def bedrock_chat_model(
    model_id: str | None = None,
    region: str | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
):
    """Construct a cached ``ChatBedrockConverse`` for the given model/region.

    Returns a LangChain-compatible chat model that ``create_react_agent``
    and ``LangGraphAgent`` can consume directly. Cached so repeat calls
    in the same process reuse the underlying client.
    """
    from langchain_aws import ChatBedrockConverse  # type: ignore[import-not-found]

    return ChatBedrockConverse(
        model=model_id or default_bedrock_model_id(),
        region_name=region or default_region(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
