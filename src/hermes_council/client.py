"""Lazy singleton AsyncOpenAI client with config resolution.

API key priority: COUNCIL_API_KEY > OPENROUTER_API_KEY > NOUS_API_KEY > OPENAI_API_KEY
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nousresearch/hermes-3-llama-3.1-70b"
_client = None
_json_mode_supported: Optional[bool] = None


def get_api_config() -> Dict[str, str]:
    """Resolve API key and base URL from environment variables."""
    if os.getenv("COUNCIL_API_KEY"):
        return {
            "api_key": os.environ["COUNCIL_API_KEY"],
            "base_url": os.getenv("COUNCIL_BASE_URL", "https://openrouter.ai/api/v1"),
        }
    if os.getenv("OPENROUTER_API_KEY"):
        return {
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
        }
    if os.getenv("NOUS_API_KEY"):
        return {
            "api_key": os.environ["NOUS_API_KEY"],
            "base_url": "https://inference-api.nousresearch.com/v1",
        }
    if os.getenv("OPENAI_API_KEY"):
        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
    return {}


def get_model() -> str:
    """Get the council model from env or use default."""
    return os.getenv("COUNCIL_MODEL", _DEFAULT_MODEL)


def get_timeout() -> float:
    """Get per-call timeout in seconds."""
    return float(os.getenv("COUNCIL_TIMEOUT", "60"))


def get_client():
    """Get or create the lazy singleton AsyncOpenAI client."""
    global _client
    if _client is not None:
        return _client

    from openai import AsyncOpenAI

    config = get_api_config()
    if not config:
        return None

    _client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=get_timeout(),
    )
    return _client


def reset_client():
    """Reset the singleton client (for testing or reconfiguration)."""
    global _client, _json_mode_supported
    _client = None
    _json_mode_supported = None


def is_json_mode_supported() -> Optional[bool]:
    """Check if JSON mode has been tested. None = untested."""
    return _json_mode_supported


def set_json_mode_supported(supported: bool):
    """Record whether JSON mode is supported by the current provider."""
    global _json_mode_supported
    _json_mode_supported = supported
    if not supported:
        logger.warning("JSON mode not supported by provider, falling back to text parsing")
