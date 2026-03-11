"""Tests for CouncilEvaluator."""

import json
from unittest.mock import AsyncMock, patch

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
    def _make_evaluator(self):
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        ev._personas = {}
        ev.model = "test"
        return ev

    def test_normal_range(self):
        ev = self._make_evaluator()
        verdict = CouncilVerdict("q", {}, "", 75, False)
        assert ev.normalized_reward(verdict) == 0.75

    def test_clamped_high(self):
        ev = self._make_evaluator()
        verdict = CouncilVerdict("q", {}, "", 150, False)
        assert ev.normalized_reward(verdict) == 1.0

    def test_clamped_low(self):
        ev = self._make_evaluator()
        verdict = CouncilVerdict("q", {}, "", -10, False)
        assert ev.normalized_reward(verdict) == 0.0


class TestDictMutationSafety:
    @pytest.mark.asyncio
    async def test_personas_not_mutated_after_evaluate(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator()
        original_keys = set(ev._personas.keys())

        mock_verdict = CouncilVerdict("q", {}, "synthesis", 75, False)
        mock_meta = {"calls_made": 5, "model": "test", "total_tokens": 0}

        with patch("hermes_council.rl.evaluator._run_council", new_callable=AsyncMock, return_value=(mock_verdict, mock_meta)):
            await ev.evaluate("test content")

        assert set(ev._personas.keys()) == original_keys


class TestExtractDPOPairs:
    def test_from_verdict(self):
        ev = CouncilEvaluator.__new__(CouncilEvaluator)
        ev._personas = {}
        ev.model = "test"

        responses = {
            "advocate": PersonaResponse("advocate", "For", 0.9, False, [], []),
            "skeptic": PersonaResponse("skeptic", "Against", 0.3, True, [], []),
            "arbiter": PersonaResponse("arbiter", "Synthesis", 0.75, False, [], []),
        }
        verdict = CouncilVerdict("Question?", responses, "Synthesis", 75, True, [], [])
        pairs = ev.extract_dpo_pairs(verdict)
        assert len(pairs) >= 1


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_returns_verdict(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator()

        mock_verdict = CouncilVerdict("q", {"arbiter": PersonaResponse("arbiter", "Good", 0.8, False, [], [])}, "Good work", 80, False)
        mock_meta = {"calls_made": 5, "model": "test", "total_tokens": 500}

        with patch("hermes_council.rl.evaluator._run_council", new_callable=AsyncMock, return_value=(mock_verdict, mock_meta)):
            result = await ev.evaluate("Some content to evaluate")
            assert isinstance(result, CouncilVerdict)
            assert result.confidence_score == 80

    @pytest.mark.asyncio
    async def test_evaluate_handles_failure(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        ev = CouncilEvaluator()

        with patch("hermes_council.rl.evaluator._run_council", new_callable=AsyncMock, return_value=(None, {"error": "fail"})):
            result = await ev.evaluate("content")
            assert isinstance(result, CouncilVerdict)
            assert result.confidence_score == 0
