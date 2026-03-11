# hermes-council MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone MCP server package that exposes the adversarial council as 3 tools (council_query, council_evaluate, council_gate) for hermes-agent.

**Architecture:** FastMCP stdio server with lazy singleton AsyncOpenAI client, JSON mode structured output with regex fallback, Pydantic validation, and an optional CouncilEvaluator for RL use. OuroborosEnv ships as an example template.

**Tech Stack:** Python 3.11+, FastMCP (mcp>=1.2.0), openai>=1.6.0, pydantic>=2.0, pyyaml, pytest, pytest-asyncio

**Design doc:** `docs/plans/2026-03-11-mcp-server-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/hermes_council/__init__.py`
- Create: `config.example.yaml`
- Create: `LICENSE`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "hermes-council"
version = "0.1.0"
description = "Adversarial multi-perspective council MCP server for hermes-agent"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    "mcp>=1.2.0",
    "openai>=1.6.0",
    "pydantic>=2.0",
    "pyyaml",
]

[project.optional-dependencies]
rl = ["openai>=1.6.0", "pyyaml", "pydantic>=2.0"]
dev = ["pytest", "pytest-asyncio"]

[project.scripts]
hermes-council-server = "hermes_council.server:main"
hermes-council = "hermes_council.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
hermes_council = ["../../skills/**/*.md"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create src/hermes_council/__init__.py**

```python
"""hermes-council: Adversarial multi-perspective council MCP server."""

__version__ = "0.1.0"
```

**Step 3: Create config.example.yaml**

```yaml
# hermes-council custom persona configuration
# Place at ~/.hermes-council/config.yaml or set COUNCIL_CONFIG env var

personas:
  # Example: add a custom "researcher" persona
  # researcher:
  #   tradition: "Systematic literature review"
  #   system_prompt: |
  #     You are the Researcher on an adversarial deliberation council.
  #     Your role is to find and synthesize published evidence...
  #     You MUST respond in valid JSON with these exact keys:
  #     {"reasoning": "...", "confidence": 0.0-1.0, "dissent": true/false,
  #      "key_points": [...], "sources": [...]}
  #     Do not include any text outside the JSON object.
  #   scoring_weights:
  #     evidence: 0.4
  #     methodology: 0.3
  #     reproducibility: 0.2
  #     novelty: 0.1
  #   tags: ["research", "systematic", "evidence"]
```

**Step 4: Create LICENSE**

MIT license, copyright 2026.

**Step 5: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
.pytest_cache/
```

**Step 6: Create directory structure**

```bash
mkdir -p src/hermes_council/rl tests examples skills/council/multi-perspective-analysis skills/council/bayesian-synthesis skills/council/adversarial-critique
```

**Step 7: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: installs successfully, `python -c "import hermes_council; print(hermes_council.__version__)"` prints `0.1.0`

**Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure with pyproject.toml"
```

---

### Task 2: Schemas (`schemas.py`)

TDD — schemas first since everything depends on them.

**Files:**
- Create: `src/hermes_council/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write failing tests**

```python
"""Tests for Pydantic response schemas."""

import pytest
from pydantic import ValidationError

from hermes_council.schemas import ArbiterOutput, DPOPair, PersonaOutput


class TestPersonaOutput:
    def test_valid_full(self):
        out = PersonaOutput(
            reasoning="Strong evidence supports this.",
            confidence=0.85,
            dissent=True,
            key_points=["point 1", "point 2"],
            sources=["https://example.com"],
        )
        assert out.confidence == 0.85
        assert out.dissent is True
        assert len(out.key_points) == 2

    def test_valid_minimal(self):
        out = PersonaOutput(reasoning="Analysis here.")
        assert out.confidence == 0.5
        assert out.dissent is False
        assert out.key_points == []
        assert out.sources == []

    def test_confidence_clamped_low(self):
        with pytest.raises(ValidationError):
            PersonaOutput(reasoning="x", confidence=-0.1)

    def test_confidence_clamped_high(self):
        with pytest.raises(ValidationError):
            PersonaOutput(reasoning="x", confidence=1.1)

    def test_from_json_string(self):
        import json

        raw = json.dumps(
            {
                "reasoning": "test",
                "confidence": 0.7,
                "dissent": False,
                "key_points": ["a"],
                "sources": [],
            }
        )
        out = PersonaOutput.model_validate_json(raw)
        assert out.confidence == 0.7
        assert out.key_points == ["a"]

    def test_from_json_extra_fields_ignored(self):
        import json

        raw = json.dumps(
            {"reasoning": "test", "confidence": 0.5, "extra_field": "ignored"}
        )
        out = PersonaOutput.model_validate_json(raw)
        assert out.reasoning == "test"


class TestArbiterOutput:
    def test_valid_full(self):
        out = ArbiterOutput(
            reasoning="Synthesis of all views.",
            confidence=0.78,
            prior="60% likely",
            posterior="78% likely",
            evidence_updates=["Advocate: +10%", "Skeptic: -5%"],
            risk_level="medium",
            consensus="Proceed with caution.",
        )
        assert out.prior == "60% likely"
        assert out.risk_level == "medium"

    def test_inherits_persona_fields(self):
        out = ArbiterOutput(reasoning="test")
        assert hasattr(out, "confidence")
        assert hasattr(out, "dissent")
        assert hasattr(out, "key_points")

    def test_defaults(self):
        out = ArbiterOutput(reasoning="test")
        assert out.prior == ""
        assert out.posterior == ""
        assert out.evidence_updates == []
        assert out.risk_level == "medium"
        assert out.consensus == ""


class TestDPOPair:
    def test_valid(self):
        pair = DPOPair(
            question="Is X true?",
            chosen="Yes because...",
            rejected="No because...",
            confidence=0.8,
            chosen_persona="arbiter",
            rejected_persona="skeptic",
        )
        assert pair.source == "council_evaluation"

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            DPOPair(question="test", chosen="yes")
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_council.schemas'`

**Step 3: Implement schemas.py**

```python
"""Pydantic models for structured JSON mode responses."""

from pydantic import BaseModel, Field


class PersonaOutput(BaseModel):
    """Structured output from a deliberator persona."""

    model_config = {"extra": "ignore"}

    reasoning: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    dissent: bool = False
    key_points: list[str] = []
    sources: list[str] = []


class ArbiterOutput(PersonaOutput):
    """Structured output from the Arbiter persona (extends PersonaOutput)."""

    prior: str = ""
    posterior: str = ""
    evidence_updates: list[str] = []
    risk_level: str = "medium"
    consensus: str = ""


class DPOPair(BaseModel):
    """A DPO preference pair extracted from council disagreement."""

    question: str
    chosen: str
    rejected: str
    confidence: float
    source: str = "council_evaluation"
    chosen_persona: str
    rejected_persona: str
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_schemas.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic schemas for structured output"
```

---

### Task 3: Personas (`personas.py`)

**Files:**
- Create: `src/hermes_council/personas.py`
- Create: `tests/test_personas.py`

**Step 1: Write failing tests**

```python
"""Tests for persona definitions and loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from hermes_council.personas import (
    CouncilVerdict,
    DEFAULT_PERSONAS,
    Persona,
    PersonaResponse,
    get_persona,
    list_personas,
    load_custom_personas,
)


class TestPersonaDataclass:
    def test_persona_fields(self):
        p = DEFAULT_PERSONAS["advocate"]
        assert p.name == "advocate"
        assert p.tradition == "Steel-manning"
        assert isinstance(p.system_prompt, str)
        assert isinstance(p.scoring_weights, dict)
        assert isinstance(p.tags, list)

    def test_scoring_weights_sum(self):
        for name, persona in DEFAULT_PERSONAS.items():
            total = sum(persona.scoring_weights.values())
            assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"

    def test_all_five_defaults(self):
        expected = {"advocate", "skeptic", "oracle", "contrarian", "arbiter"}
        assert set(DEFAULT_PERSONAS.keys()) == expected


class TestPersonaResponse:
    def test_construction(self):
        r = PersonaResponse(
            persona_name="skeptic",
            content="Analysis text",
            confidence=0.7,
            dissents=True,
            key_points=["flaw 1"],
            sources=["https://example.com"],
        )
        assert r.dissents is True
        assert r.confidence == 0.7


class TestCouncilVerdict:
    def test_construction(self):
        v = CouncilVerdict(
            question="Is X true?",
            responses={},
            arbiter_synthesis="Yes with caveats",
            confidence_score=75,
            conflict_detected=False,
        )
        assert v.confidence_score == 75
        assert v.dpo_pairs == []
        assert v.sources == []


class TestGetPersona:
    def test_case_insensitive(self):
        assert get_persona("ADVOCATE") is not None
        assert get_persona("Skeptic") is not None
        assert get_persona("arbiter") is not None

    def test_not_found(self):
        assert get_persona("nonexistent") is None


class TestListPersonas:
    def test_returns_all_names(self):
        names = list_personas()
        assert len(names) == 5
        assert "arbiter" in names


class TestJsonInstructions:
    def test_deliberator_prompts_request_json(self):
        for name in ["advocate", "skeptic", "oracle", "contrarian"]:
            prompt = DEFAULT_PERSONAS[name].system_prompt
            assert "JSON" in prompt
            assert '"reasoning"' in prompt
            assert '"confidence"' in prompt

    def test_arbiter_prompt_has_extra_keys(self):
        prompt = DEFAULT_PERSONAS["arbiter"].system_prompt
        assert '"prior"' in prompt
        assert '"posterior"' in prompt
        assert '"consensus"' in prompt


class TestCustomPersonaLoading:
    def test_defaults_when_no_config(self):
        personas = load_custom_personas("/nonexistent/path.yaml")
        assert len(personas) == 5

    def test_custom_merges_with_defaults(self):
        config = {
            "personas": {
                "researcher": {
                    "tradition": "Systematic review",
                    "system_prompt": "You are the Researcher...",
                    "scoring_weights": {"evidence": 0.5, "rigor": 0.5},
                    "tags": ["research"],
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            f.flush()
            try:
                personas = load_custom_personas(f.name)
                assert "researcher" in personas
                assert "advocate" in personas  # defaults still present
                assert len(personas) == 6
            finally:
                os.unlink(f.name)

    def test_custom_overrides_default(self):
        config = {
            "personas": {
                "advocate": {
                    "tradition": "Custom tradition",
                    "system_prompt": "Custom prompt",
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            f.flush()
            try:
                personas = load_custom_personas(f.name)
                assert personas["advocate"].tradition == "Custom tradition"
                assert len(personas) == 5
            finally:
                os.unlink(f.name)

    def test_env_var_config_path(self, monkeypatch, tmp_path):
        config_file = tmp_path / "council.yaml"
        config_file.write_text(yaml.dump({"personas": {}}))
        monkeypatch.setenv("COUNCIL_CONFIG", str(config_file))
        personas = load_custom_personas()
        assert len(personas) == 5
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_personas.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement personas.py**

Port the original `tools/council_personas.py` from the PR with these changes:
- All 5 system prompts get JSON output instructions appended (deliberator format for 4, arbiter format for arbiter)
- `load_custom_personas()` reads from `COUNCIL_CONFIG` env var or `~/.hermes-council/config.yaml` (NOT `~/.hermes/config.yaml`)
- Config YAML key is `personas` at root (not nested under `council.personas`)
- All dataclasses, scoring_weights, and tags preserved exactly

The full system prompts are long — carry them over from the original PR source verbatim, then append the JSON instruction block to each.

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_personas.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/personas.py tests/test_personas.py
git commit -m "feat: add persona definitions with JSON output instructions"
```

---

### Task 4: Regex Fallback Parsers (`parsing.py`)

**Files:**
- Create: `src/hermes_council/parsing.py`
- Create: `tests/test_parsing.py`

**Step 1: Write failing tests**

```python
"""Tests for regex fallback parsers."""

from hermes_council.parsing import (
    parse_confidence,
    parse_dissent,
    parse_key_points,
    extract_sources,
    parse_persona_response,
)
from hermes_council.personas import PersonaResponse


class TestParseConfidence:
    def test_decimal(self):
        assert parse_confidence("CONFIDENCE: 0.85") == 0.85

    def test_percentage(self):
        assert parse_confidence("CONFIDENCE: 85%") == 0.85

    def test_integer_over_one(self):
        assert parse_confidence("Confidence: 72") == 0.72

    def test_missing(self):
        assert parse_confidence("No confidence here") == 0.5


class TestParseDissent:
    def test_true(self):
        assert parse_dissent("DISSENT: true") is True

    def test_false(self):
        assert parse_dissent("DISSENT: false") is False

    def test_case_insensitive(self):
        assert parse_dissent("Dissent: True") is True

    def test_missing(self):
        assert parse_dissent("No dissent field") is False


class TestParseKeyPoints:
    def test_bullet_points(self):
        text = "Header\n- First important point here\n- Second point here\n- Short"
        points = parse_key_points(text)
        assert len(points) == 2  # "Short" is <10 chars, skipped

    def test_asterisk_bullets(self):
        text = "* A significant finding\n* Another finding here"
        points = parse_key_points(text)
        assert len(points) == 2

    def test_max_ten(self):
        text = "\n".join(f"- Point number {i} with enough text" for i in range(15))
        points = parse_key_points(text)
        assert len(points) == 10


class TestExtractSources:
    def test_urls(self):
        text = "See https://example.com and http://test.org/page for details."
        sources = extract_sources(text)
        assert len(sources) == 2

    def test_deduplication(self):
        text = "https://example.com and again https://example.com"
        sources = extract_sources(text)
        assert len(sources) == 1

    def test_no_urls(self):
        assert extract_sources("No links here") == []


class TestParsePersonaResponse:
    def test_full_parse(self):
        text = (
            "Analysis here.\n"
            "- Key point one for analysis\n"
            "- Key point two for analysis\n"
            "CONFIDENCE: 0.8\n"
            "DISSENT: true\n"
            "Source: https://example.com"
        )
        resp = parse_persona_response("skeptic", text)
        assert isinstance(resp, PersonaResponse)
        assert resp.persona_name == "skeptic"
        assert resp.confidence == 0.8
        assert resp.dissents is True
        assert len(resp.key_points) == 2
        assert len(resp.sources) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_parsing.py -v`
Expected: FAIL

**Step 3: Implement parsing.py**

```python
"""Regex fallback parsers for non-JSON-mode providers.

These are used when the LLM provider does not support
response_format={"type": "json_object"}. Ported from the original
hermes-agent PR #848 council_tool.py.
"""

import re
from typing import List

from hermes_council.personas import PersonaResponse


def parse_confidence(text: str) -> float:
    """Extract confidence value from persona response text."""
    patterns = [
        r"CONFIDENCE:\s*([\d.]+)",
        r"confidence[:\s]+([\d.]+)",
        r"(\d+)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
    return 0.5


def parse_dissent(text: str) -> bool:
    """Extract dissent flag from persona response text."""
    match = re.search(r"DISSENT:\s*(true|false)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return False


def parse_key_points(text: str) -> List[str]:
    """Extract bullet points from persona response text."""
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "  - ", "  * ")):
            point = line.lstrip("-* ").strip()
            if len(point) > 10:
                points.append(point)
    return points[:10]


def extract_sources(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
    return list(set(re.findall(url_pattern, text)))


def parse_persona_response(persona_name: str, raw_text: str) -> PersonaResponse:
    """Parse raw LLM output into a structured PersonaResponse using regex."""
    return PersonaResponse(
        persona_name=persona_name,
        content=raw_text,
        confidence=parse_confidence(raw_text),
        dissents=parse_dissent(raw_text),
        key_points=parse_key_points(raw_text),
        sources=extract_sources(raw_text),
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_parsing.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/parsing.py tests/test_parsing.py
git commit -m "feat: add regex fallback parsers for non-JSON-mode providers"
```

---

### Task 5: Client (`client.py`)

**Files:**
- Create: `src/hermes_council/client.py`
- Create: `tests/test_client.py`

**Step 1: Write failing tests**

```python
"""Tests for API client configuration and lazy singleton."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_council.client import get_api_config, get_client, get_model, reset_client


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
        for key in [
            "COUNCIL_API_KEY",
            "OPENROUTER_API_KEY",
            "NOUS_API_KEY",
            "OPENAI_API_KEY",
        ]:
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


class TestLazySingleton:
    def test_reset_clears_client(self, monkeypatch):
        monkeypatch.setenv("COUNCIL_API_KEY", "test")
        reset_client()
        # After reset, next get_client() creates a new instance
        # (we just verify reset doesn't crash)
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_client.py -v`
Expected: FAIL

**Step 3: Implement client.py**

```python
"""Lazy singleton AsyncOpenAI client with config resolution.

API key priority: COUNCIL_API_KEY > OPENROUTER_API_KEY > NOUS_API_KEY > OPENAI_API_KEY
"""

import logging
import os
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nousresearch/hermes-3-llama-3.1-70b"
_client = None
_json_mode_supported: Optional[bool] = None  # None = untested, True/False = tested


def get_api_config() -> Dict[str, str]:
    """Resolve API key and base URL from environment variables."""
    if os.getenv("COUNCIL_API_KEY"):
        return {
            "api_key": os.environ["COUNCIL_API_KEY"],
            "base_url": os.getenv(
                "COUNCIL_BASE_URL", "https://openrouter.ai/api/v1"
            ),
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
        logger.warning(
            "JSON mode not supported by provider, falling back to text parsing"
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_client.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/client.py tests/test_client.py
git commit -m "feat: add lazy singleton client with config resolution"
```

---

### Task 6: Deliberation Engine (`deliberation.py`)

The core logic. Depends on schemas, personas, client, parsing.

**Files:**
- Create: `src/hermes_council/deliberation.py`
- Create: `tests/test_deliberation.py`

**Step 1: Write failing tests**

```python
"""Tests for core deliberation logic."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_council.deliberation import (
    _extract_dpo_pairs,
    _run_council,
    _run_gate,
    llm_call,
)
from hermes_council.personas import CouncilVerdict, PersonaResponse


def _mock_persona_json(confidence=0.8, dissent=False):
    """Return a valid JSON string matching PersonaOutput schema."""
    return json.dumps(
        {
            "reasoning": "Test analysis.",
            "confidence": confidence,
            "dissent": dissent,
            "key_points": ["point 1"],
            "sources": ["https://example.com"],
        }
    )


def _mock_arbiter_json(confidence=0.75):
    """Return a valid JSON string matching ArbiterOutput schema."""
    return json.dumps(
        {
            "reasoning": "Synthesis of all views.",
            "confidence": confidence,
            "dissent": False,
            "key_points": ["synthesis point"],
            "sources": [],
            "prior": "60%",
            "posterior": "75%",
            "evidence_updates": ["Advocate: +10%", "Skeptic: -5%"],
            "risk_level": "medium",
            "consensus": "Proceed with caution.",
        }
    )


class TestLLMCall:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        with patch("hermes_council.deliberation.get_client", return_value=None):
            result, tokens = await llm_call("system", "user")
            assert "No API key" in result
            assert tokens == 0

    @pytest.mark.asyncio
    async def test_successful_json_mode(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"reasoning": "test"}'
        mock_response.usage = MagicMock(total_tokens=150)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("hermes_council.deliberation.get_client", return_value=mock_client),
            patch("hermes_council.deliberation.is_json_mode_supported", return_value=True),
        ):
            result, tokens = await llm_call("system", "user")
            assert result == '{"reasoning": "test"}'
            assert tokens == 150


class TestRunCouncil:
    @pytest.mark.asyncio
    async def test_full_deliberation(self):
        call_count = 0

        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                return _mock_persona_json(confidence=0.8), 100
            return _mock_arbiter_json(), 200

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_council("Is X true?")
            assert isinstance(verdict, CouncilVerdict)
            assert verdict.confidence_score > 0
            assert meta["calls_made"] == 5
            assert meta["total_tokens"] == 600  # 4*100 + 200

    @pytest.mark.asyncio
    async def test_persona_subset(self):
        call_count = 0

        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _mock_persona_json(), 100
            return _mock_arbiter_json(), 200

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_council(
                "Test?", persona_names=["skeptic", "arbiter"]
            )
            assert meta["calls_made"] == 2

    @pytest.mark.asyncio
    async def test_one_deliberator_fails(self):
        call_count = 0

        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")
            if call_count <= 4:
                return _mock_persona_json(), 100
            return _mock_arbiter_json(), 200

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_council("Test?")
            assert isinstance(verdict, CouncilVerdict)
            assert len(verdict.responses) >= 3  # 3 succeeded + arbiter

    @pytest.mark.asyncio
    async def test_all_deliberators_fail(self):
        async def mock_llm(system, user, model=None):
            raise Exception("All fail")

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_council("Test?")
            assert verdict is None
            assert "error" in meta

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        call_count = 0

        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_persona_json(confidence=0.9), 100
            if call_count == 2:
                return _mock_persona_json(confidence=0.3, dissent=True), 100
            if call_count <= 4:
                return _mock_persona_json(confidence=0.6), 100
            return _mock_arbiter_json(), 200

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_council("Controversial?")
            assert verdict.conflict_detected is True

    @pytest.mark.asyncio
    async def test_evidence_search_flag(self):
        captured_messages = []

        async def mock_llm(system, user, model=None):
            captured_messages.append(user)
            return _mock_persona_json(), 100

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            await _run_council("Test?", evidence_search=True)
            assert any("web search" in msg for msg in captured_messages)

            captured_messages.clear()
            await _run_council("Test?", evidence_search=False)
            assert not any("web search" in msg for msg in captured_messages)


class TestRunGate:
    @pytest.mark.asyncio
    async def test_gate_uses_three_personas(self):
        call_count = 0

        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_persona_json(), 100
            return _mock_arbiter_json(), 200

        with patch("hermes_council.deliberation.llm_call", side_effect=mock_llm):
            verdict, meta = await _run_gate("Deploy code")
            assert meta["calls_made"] == 3


class TestDPOExtraction:
    def test_with_dissenter(self):
        responses = {
            "advocate": PersonaResponse("advocate", "For", 0.9, False, [], []),
            "skeptic": PersonaResponse("skeptic", "Against", 0.3, True, [], []),
            "arbiter": PersonaResponse("arbiter", "Synthesis", 0.75, False, [], []),
        }
        pairs = _extract_dpo_pairs("Question?", responses)
        assert len(pairs) >= 1
        assert pairs[0]["chosen_persona"] == "arbiter"
        assert pairs[0]["rejected_persona"] == "skeptic"

    def test_no_dissenters(self):
        responses = {
            "advocate": PersonaResponse("advocate", "For", 0.9, False, [], []),
            "oracle": PersonaResponse("oracle", "Data says yes", 0.8, False, [], []),
            "arbiter": PersonaResponse("arbiter", "Agreed", 0.85, False, [], []),
        }
        pairs = _extract_dpo_pairs("Question?", responses)
        assert len(pairs) == 0

    def test_empty_responses(self):
        pairs = _extract_dpo_pairs("Question?", {})
        assert pairs == []
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_deliberation.py -v`
Expected: FAIL

**Step 3: Implement deliberation.py**

The core module. Key functions:
- `llm_call(system_prompt, user_message, model=None)` → `(str, int)` — makes one LLM call, returns content and token count. Tries JSON mode first, falls back on 400 error, uses regex fallback via `parsing.py`.
- `_build_persona_response(persona_name, raw_text, is_arbiter=False)` → `PersonaResponse` — parses JSON (or regex fallback) into PersonaResponse.
- `_run_council(question, context, persona_names, evidence_search, model)` → `(CouncilVerdict | None, dict)` — full deliberation. Returns (None, {"error": ...}) if all deliberators fail.
- `_run_gate(action, risk_level, context)` → `(CouncilVerdict | None, dict)` — abbreviated council.
- `_extract_dpo_pairs(question, responses)` → `list[dict]` — ported from original.

Implementation details:
- Import `get_client`, `get_model`, `is_json_mode_supported`, `set_json_mode_supported` from `client`
- Import `PersonaOutput`, `ArbiterOutput` from `schemas`
- Import `parse_persona_response` from `parsing` for fallback
- Import `load_custom_personas`, `DEFAULT_PERSONAS` from `personas`
- Truncate each deliberator response to 3000 chars when building Arbiter context
- Track `total_tokens` across all calls
- Use `asyncio.gather(*tasks, return_exceptions=True)` for parallel deliberators

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_deliberation.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/deliberation.py tests/test_deliberation.py
git commit -m "feat: add core deliberation engine with JSON mode + regex fallback"
```

---

### Task 7: MCP Server (`server.py`)

**Files:**
- Create: `src/hermes_council/server.py`
- Create: `tests/test_server.py`

**Step 1: Write failing tests**

```python
"""Tests for FastMCP server tool registration."""

import json
from unittest.mock import AsyncMock, patch

import pytest


class TestServerTools:
    def test_server_has_three_tools(self):
        from hermes_council.server import mcp

        tools = mcp._tool_manager._tools
        assert "council_query" in tools
        assert "council_evaluate" in tools
        assert "council_gate" in tools

    def test_council_query_missing_question(self):
        from hermes_council.server import _council_query

        # Call without required param
        result = asyncio.get_event_loop().run_until_complete(
            _council_query(question="")
        )
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "question" in parsed["error"].lower()

    def test_council_evaluate_missing_content(self):
        from hermes_council.server import _council_evaluate

        result = asyncio.get_event_loop().run_until_complete(
            _council_evaluate(content="")
        )
        parsed = json.loads(result)
        assert parsed["success"] is False

    def test_council_gate_missing_action(self):
        from hermes_council.server import _council_gate

        result = asyncio.get_event_loop().run_until_complete(
            _council_gate(action="")
        )
        parsed = json.loads(result)
        assert parsed["success"] is False


class TestServerResponses:
    @pytest.mark.asyncio
    async def test_query_response_has_meta(self):
        from hermes_council.personas import CouncilVerdict, PersonaResponse
        from hermes_council.server import _council_query

        mock_verdict = CouncilVerdict(
            question="Test?",
            responses={
                "arbiter": PersonaResponse("arbiter", "Synthesis", 0.8, False, ["p"], [])
            },
            arbiter_synthesis="Synthesis text",
            confidence_score=80,
            conflict_detected=False,
            dpo_pairs=[],
            sources=[],
        )
        mock_meta = {"calls_made": 5, "model": "test-model", "total_tokens": 500}

        with patch(
            "hermes_council.server._run_council",
            new_callable=AsyncMock,
            return_value=(mock_verdict, mock_meta),
        ):
            result = await _council_query(question="Test?")
            parsed = json.loads(result)
            assert parsed["success"] is True
            assert parsed["_meta"]["calls_made"] == 5
            assert "available_personas" in parsed

    @pytest.mark.asyncio
    async def test_gate_response_has_allowed(self):
        from hermes_council.personas import CouncilVerdict, PersonaResponse
        from hermes_council.server import _council_gate

        mock_verdict = CouncilVerdict(
            question="Gate test",
            responses={
                "skeptic": PersonaResponse("skeptic", "Concerns", 0.6, True, ["risk"], []),
                "arbiter": PersonaResponse("arbiter", "OK", 0.7, False, [], []),
            },
            arbiter_synthesis="Proceed",
            confidence_score=70,
            conflict_detected=False,
        )
        mock_meta = {"calls_made": 3, "model": "test", "total_tokens": 300}

        with patch(
            "hermes_council.server._run_council",
            new_callable=AsyncMock,
            return_value=(mock_verdict, mock_meta),
        ):
            result = await _council_gate(action="Deploy code", risk_level="medium")
            parsed = json.loads(result)
            assert "allowed" in parsed
            assert parsed["threshold"] == 50
```

Note: These tests import the server module directly to test the handler functions, not through MCP protocol. Adjust imports as needed based on how FastMCP exposes registered tools.

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_server.py -v`
Expected: FAIL

**Step 3: Implement server.py**

```python
"""FastMCP stdio server exposing council tools."""

import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from hermes_council.deliberation import _run_council, _run_gate
from hermes_council.personas import PersonaResponse, list_personas

# All logging to stderr — stdout is the MCP JSON-RPC protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hermes_council")

mcp = FastMCP("hermes-council")


@mcp.tool()
async def council_query(
    question: str,
    context: str = "",
    personas: Optional[list[str]] = None,
    evidence_search: bool = True,
) -> str:
    """Submit a question for 5-persona adversarial deliberation. Returns structured verdict with confidence score, evidence links, and DPO pairs."""
    # ... validate, call _run_council, format response with _meta
    pass


@mcp.tool()
async def council_evaluate(
    content: str,
    question: str = "",
    criteria: Optional[list[str]] = None,
) -> str:
    """Evaluate content quality through adversarial council critique. Returns confidence score and structured feedback."""
    # ... validate, call _run_council with eval framing, format response
    pass


@mcp.tool()
async def council_gate(
    action: str,
    risk_level: str = "medium",
    context: str = "",
) -> str:
    """Quick safety review before high-stakes actions. Uses Skeptic + Oracle + Arbiter. Returns allow/deny with reasoning."""
    # ... validate, call _run_gate, apply threshold, format response
    pass


def main():
    """Entry point for hermes-council-server."""
    mcp.run(transport="stdio")
```

Fill in all three handlers following the design doc response formats exactly. Each handler:
1. Validates required params (return `{"success": false, "error": "..."}` if missing)
2. Calls `_run_council` or `_run_gate`
3. Handles `verdict is None` (all-fail case)
4. Formats response JSON with persona_responses (truncated), dpo_pairs, sources, `_meta`
5. `council_query` includes `available_personas` in response
6. `council_gate` computes `allowed` from confidence vs threshold

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_server.py -v`
Expected: all PASS

**Step 5: Verify server starts**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && echo '{}' | timeout 3 hermes-council-server 2>/dev/null; echo "exit: $?"`
Expected: starts and exits cleanly (no crash on empty input)

**Step 6: Commit**

```bash
git add src/hermes_council/server.py tests/test_server.py
git commit -m "feat: add FastMCP stdio server with 3 council tools"
```

---

### Task 8: RL Evaluator (`rl/evaluator.py`)

**Files:**
- Create: `src/hermes_council/rl/__init__.py`
- Create: `src/hermes_council/rl/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write failing tests**

```python
"""Tests for CouncilEvaluator (standalone RL component)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_council.personas import CouncilVerdict, PersonaResponse
from hermes_council.rl.evaluator import CouncilEvaluator


class TestEvaluatorInit:
    def test_default_config(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator()
        assert ev.model == "nousresearch/hermes-3-llama-3.1-70b"
        assert len(ev._personas) == 5

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator(model="custom/model")
        assert ev.model == "custom/model"

    def test_persona_subset(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator(personas=["skeptic", "arbiter"])
        assert "skeptic" in ev._personas
        assert "arbiter" in ev._personas
        assert "advocate" not in ev._personas


class TestNormalizedReward:
    def test_normal_range(self):
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        verdict = CouncilVerdict("q", {}, "", 75, False)
        assert ev.normalized_reward(verdict) == 0.75

    def test_clamped_high(self):
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        verdict = CouncilVerdict("q", {}, "", 150, False)
        assert ev.normalized_reward(verdict) == 1.0

    def test_clamped_low(self):
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        verdict = CouncilVerdict("q", {}, "", -10, False)
        assert ev.normalized_reward(verdict) == 0.0


class TestDictMutationSafety:
    def test_personas_not_mutated_after_evaluate(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator()
        original_keys = set(ev._personas.keys())

        mock_verdict = CouncilVerdict("q", {}, "synthesis", 75, False)
        mock_meta = {"calls_made": 5, "model": "test", "total_tokens": 0}

        with patch(
            "hermes_council.rl.evaluator._run_council",
            new_callable=AsyncMock,
            return_value=(mock_verdict, mock_meta),
        ):
            asyncio.get_event_loop().run_until_complete(
                ev.evaluate("test content")
            )

        assert set(ev._personas.keys()) == original_keys


class TestExtractDPOPairs:
    def test_from_verdict(self):
        responses = {
            "advocate": PersonaResponse("advocate", "For", 0.9, False, [], []),
            "skeptic": PersonaResponse("skeptic", "Against", 0.3, True, [], []),
            "arbiter": PersonaResponse("arbiter", "Synthesis", 0.75, False, [], []),
        }
        verdict = CouncilVerdict(
            "Question?", responses, "Synthesis", 75, True, [], []
        )
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        pairs = ev.extract_dpo_pairs(verdict)
        assert len(pairs) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_evaluator.py -v`
Expected: FAIL

**Step 3: Implement evaluator.py**

Port the original `environments/council_evaluator.py` with these changes:
- Import from `hermes_council.personas` (not `tools.council_personas`)
- Import `_run_council` from `hermes_council.deliberation` instead of reimplementing LLM calls
- `evaluate()` copies `self._personas` with `dict(self._personas)` before passing to `_run_council`
- Own lazy client (calls `_run_council` which uses the `client.py` singleton, but evaluator can also be used standalone with explicit api_key/base_url params that override)
- `_run_council` is called directly, not through MCP

Create `src/hermes_council/rl/__init__.py`:
```python
"""RL components for hermes-council."""
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/test_evaluator.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hermes_council/rl/ tests/test_evaluator.py
git commit -m "feat: add CouncilEvaluator for RL integration"
```

---

### Task 9: CLI (`cli.py`)

**Files:**
- Create: `src/hermes_council/cli.py`

**Step 1: Implement cli.py**

```python
"""CLI for hermes-council: skill installation."""

import argparse
import shutil
import sys
from pathlib import Path


def _get_skills_source() -> Path:
    """Locate bundled skills directory."""
    # Skills are at repo_root/skills/council/ relative to package
    package_dir = Path(__file__).resolve().parent
    skills_dir = package_dir.parent.parent / "skills" / "council"
    if not skills_dir.exists():
        # Fallback: try importlib.resources for installed package
        try:
            import importlib.resources as resources

            ref = resources.files("hermes_council").joinpath(
                "../../skills/council"
            )
            skills_dir = Path(str(ref))
        except Exception:
            pass
    return skills_dir


def install_skills(force: bool = False):
    """Copy council skills to ~/.hermes/skills/council/."""
    source = _get_skills_source()
    if not source.exists():
        print(f"Error: Skills source not found at {source}", file=sys.stderr)
        sys.exit(1)

    target = Path.home() / ".hermes" / "skills" / "council"

    if target.exists() and not force:
        print(
            f"Skills already installed at {target}\n"
            f"Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target)
    print(f"Installed council skills to {target}")

    # List installed files
    for path in sorted(target.rglob("*.md")):
        print(f"  {path.relative_to(target)}")


def main():
    """Entry point for hermes-council CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes-council",
        description="hermes-council: Adversarial deliberation tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    install_parser = subparsers.add_parser(
        "install-skills", help="Install council skills to ~/.hermes/skills/"
    )
    install_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing skills"
    )

    args = parser.parse_args()

    if args.command == "install-skills":
        install_skills(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Verify CLI works**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m hermes_council.cli --help`
Expected: prints help with `install-skills` subcommand

**Step 3: Commit**

```bash
git add src/hermes_council/cli.py
git commit -m "feat: add CLI with install-skills command"
```

---

### Task 10: Skills (carry over from PR)

**Files:**
- Create: `skills/council/DESCRIPTION.md`
- Create: `skills/council/multi-perspective-analysis/SKILL.md`
- Create: `skills/council/bayesian-synthesis/SKILL.md`
- Create: `skills/council/adversarial-critique/SKILL.md`

**Step 1: Copy skill files**

Fetch all 4 skill markdown files from the original PR branch (`Ridwannurudeen/hermes-agent:feat/council-subsystem`) and write them unchanged to the new locations.

Use:
```bash
gh api "repos/Ridwannurudeen/hermes-agent/contents/skills/council/DESCRIPTION.md?ref=feat/council-subsystem" --jq '.content' | base64 -d > skills/council/DESCRIPTION.md
# repeat for each skill file
```

**Step 2: Verify files exist**

Run: `find skills/ -name "*.md" -type f`
Expected: 4 files listed

**Step 3: Commit**

```bash
git add skills/
git commit -m "feat: add council skills (carried over from PR #848)"
```

---

### Task 11: Examples (OuroborosEnv + datagen config)

**Files:**
- Create: `examples/ouroboros_env.py`
- Create: `examples/ouroboros.yaml`

**Step 1: Copy and adapt OuroborosEnv**

Fetch `environments/ouroboros_env.py` from original PR. Adapt the import at the top:
- Change `from environments.council_evaluator import CouncilEvaluator` → `from hermes_council.rl.evaluator import CouncilEvaluator`
- Keep all hermes-agent imports as-is (this is a template users copy into hermes-agent)
- Add a comment at the top: `# Copy this file into your hermes-agent/environments/ directory`

**Step 2: Copy datagen config**

Fetch `datagen-config-examples/ouroboros.yaml` from original PR and write to `examples/ouroboros.yaml`.

**Step 3: Commit**

```bash
git add examples/
git commit -m "feat: add OuroborosEnv example and datagen config"
```

---

### Task 12: README

**Files:**
- Create: `README.md`

**Step 1: Write README**

Sections:
1. **hermes-council** — one-line description
2. **Install** — `pip install hermes-council` + config snippet
3. **Tools** — table of 3 tools with descriptions
4. **Configuration** — env var table from design doc
5. **Custom Personas** — example YAML
6. **Skills** — `hermes-council install-skills`
7. **RL Integration** — CouncilEvaluator usage + OuroborosEnv copy instructions
8. **Development** — `pip install -e ".[dev]"`, `pytest`
9. **License** — MIT

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with installation and usage guide"
```

---

### Task 13: Full Test Suite Run + Cleanup

**Step 1: Run entire test suite**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m pytest tests/ -v --tb=short`
Expected: all tests pass, 50+ tests

**Step 2: Run linter (if ruff available)**

Run: `cd C:/Users/GUDMAN/Desktop/hermes-council && python -m ruff check src/ tests/`
Expected: clean or fix any issues

**Step 3: Verify package installs clean**

Run: `pip install -e ".[dev]" && hermes-council --help && python -c "from hermes_council.server import mcp; print('Server OK')"`
Expected: all pass

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: test suite cleanup and final fixes"
```

---

### Task 14: Push to GitHub + Comment on PR #848

**Step 1: Create GitHub repo**

```bash
gh repo create Ridwannurudeen/hermes-council --public --description "Adversarial multi-perspective council MCP server for hermes-agent" --source C:/Users/GUDMAN/Desktop/hermes-council
```

**Step 2: Push all commits**

```bash
cd C:/Users/GUDMAN/Desktop/hermes-council && git push -u origin main
```

**Step 3: Comment on PR #848**

```bash
gh pr comment 848 --repo NousResearch/hermes-agent --body "MCP server version ready for review: https://github.com/Ridwannurudeen/hermes-council

Changes from the original PR:
- Standalone FastMCP stdio server (pip install hermes-council)
- Own config via COUNCIL_* env vars (no provider bypass)
- JSON mode structured output with Pydantic validation (regex fallback for non-supporting providers)
- Cost transparency: _meta block in every response (calls_made, model, total_tokens)
- CouncilEvaluator ships as library, OuroborosEnv as example template
- 50+ tests, all mocked

Install:
\`\`\`bash
pip install git+https://github.com/Ridwannurudeen/hermes-council.git
\`\`\`

Config:
\`\`\`yaml
mcp_servers:
  council:
    command: hermes-council-server
\`\`\`"
```
