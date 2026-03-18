"""Tests for persona definitions and loading."""

import os
import tempfile

import yaml

from hermes_council.personas import (
    CouncilVerdict,
    DEFAULT_PERSONAS,
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
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, f)
        f.flush()
        f.close()
        try:
            personas = load_custom_personas(f.name)
            assert "researcher" in personas
            assert "advocate" in personas
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
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, f)
        f.flush()
        f.close()
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
