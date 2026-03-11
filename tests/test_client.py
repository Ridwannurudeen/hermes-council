"""Tests for API client configuration and lazy singleton."""

import pytest

from hermes_council.client import get_api_config, get_client, get_model, get_timeout, reset_client, is_json_mode_supported, set_json_mode_supported


class TestGetApiConfig:
    def test_council_key_highest_priority(self, monkeypatch):
        monkeypatch.setenv("COUNCIL_API_KEY", "ck_test")
        monkeypatch.setenv("COUNCIL_BASE_URL", "https://custom.api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or_test")
        config = get_api_config()
        assert config["api_key"] == "ck_test"
        assert config["base_url"] == "https://custom.api/v1"

    def test_council_key_default_base_url(self, monkeypatch):
        monkeypatch.setenv("COUNCIL_API_KEY", "ck_test")
        monkeypatch.delenv("COUNCIL_BASE_URL", raising=False)
        config = get_api_config()
        assert config["base_url"] == "https://openrouter.ai/api/v1"

    def test_openrouter_fallback(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or_test")
        config = get_api_config()
        assert config["api_key"] == "or_test"
        assert config["base_url"] == "https://openrouter.ai/api/v1"

    def test_nous_fallback(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("NOUS_API_KEY", "nous_test")
        config = get_api_config()
        assert config["api_key"] == "nous_test"
        assert config["base_url"] == "https://inference-api.nousresearch.com/v1"

    def test_openai_fallback(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
        config = get_api_config()
        assert config["api_key"] == "sk_test"
        assert config["base_url"] == "https://api.openai.com/v1"

    def test_openai_custom_base_url(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai/v1")
        config = get_api_config()
        assert config["base_url"] == "https://custom.openai/v1"

    def test_no_keys(self, monkeypatch):
        for key in ["COUNCIL_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        config = get_api_config()
        assert config == {}


class TestGetModel:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_MODEL", raising=False)
        assert get_model() == "nousresearch/hermes-3-llama-3.1-70b"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("COUNCIL_MODEL", "custom/model")
        assert get_model() == "custom/model"


class TestGetTimeout:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("COUNCIL_TIMEOUT", raising=False)
        assert get_timeout() == 60.0

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("COUNCIL_TIMEOUT", "120")
        assert get_timeout() == 120.0


class TestLazySingleton:
    def test_no_key_returns_none(self, monkeypatch):
        for key in ["COUNCIL_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        reset_client()
        assert get_client() is None

    def test_reset_clears_client(self):
        reset_client()
        # After reset, _client is None — next get_client() creates fresh


class TestJsonModeFlag:
    def test_initially_none(self):
        reset_client()
        assert is_json_mode_supported() is None

    def test_set_and_get(self):
        reset_client()
        set_json_mode_supported(True)
        assert is_json_mode_supported() is True
        set_json_mode_supported(False)
        assert is_json_mode_supported() is False
        reset_client()  # cleanup
