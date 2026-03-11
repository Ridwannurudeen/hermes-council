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
        raw = json.dumps({
            "reasoning": "test",
            "confidence": 0.7,
            "dissent": False,
            "key_points": ["a"],
            "sources": [],
        })
        out = PersonaOutput.model_validate_json(raw)
        assert out.confidence == 0.7
        assert out.key_points == ["a"]

    def test_from_json_extra_fields_ignored(self):
        import json
        raw = json.dumps({"reasoning": "test", "confidence": 0.5, "extra_field": "ignored"})
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
