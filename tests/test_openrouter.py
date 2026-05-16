"""Tests for ahp.llm.openrouter — pure config-shaping, no real API calls."""

from __future__ import annotations

import pytest

from ahp.llm import openrouter


def test_default_model_id_falls_back_when_env_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert openrouter.default_openrouter_model_id() == openrouter.OPENROUTER_DEFAULT_MODEL_ID


def test_default_model_id_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3-70b-instruct")
    assert (
        openrouter.default_openrouter_model_id()
        == "meta-llama/llama-3-70b-instruct"
    )


def test_default_base_url_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example.com/v1")
    assert openrouter.default_base_url() == "https://proxy.example.com/v1"


def test_default_base_url_falls_back(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    assert openrouter.default_base_url() == openrouter.OPENROUTER_DEFAULT_BASE_URL


def test_has_credentials_true_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert openrouter.has_openrouter_credentials() is True


def test_has_credentials_false_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert openrouter.has_openrouter_credentials() is False


def test_chat_model_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Clear the lru_cache so the test isn't served a cached client.
    openrouter.openrouter_chat_model.cache_clear()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter.openrouter_chat_model()


def test_chat_model_constructs_with_credentials(monkeypatch):
    """Verify the wiring: model id + base url + key reach ChatOpenAI."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    openrouter.openrouter_chat_model.cache_clear()

    chat = openrouter.openrouter_chat_model(
        model="anthropic/claude-3-haiku",
    )
    # ChatOpenAI stores its configuration on instance attributes.
    assert chat.model_name == "anthropic/claude-3-haiku"
    assert str(chat.openai_api_base) == openrouter.OPENROUTER_DEFAULT_BASE_URL


def test_chat_model_cached_per_model_and_base(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    openrouter.openrouter_chat_model.cache_clear()

    a = openrouter.openrouter_chat_model(model="m1")
    b = openrouter.openrouter_chat_model(model="m1")
    c = openrouter.openrouter_chat_model(model="m2")
    assert a is b
    assert a is not c
