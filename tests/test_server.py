"""Tests for FastMCP server tool handlers."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from hermes_council.personas import CouncilVerdict, PersonaResponse
from hermes_council.server import council_query, council_evaluate, council_gate


def _mock_verdict(confidence=75, conflict=False):
    return CouncilVerdict(
        question="Test?",
        responses={
            "advocate": PersonaResponse("advocate", "For the position", 0.85, False, ["strong point"], []),
            "skeptic": PersonaResponse("skeptic", "Against it", 0.4, True, ["weakness found"], []),
            "arbiter": PersonaResponse("arbiter", "Balanced view", 0.75, False, ["synthesis"], []),
        },
        arbiter_synthesis="Balanced synthesis text",
        confidence_score=confidence,
        conflict_detected=conflict,
        dpo_pairs=[],
        sources=["https://example.com"],
    )


def _mock_meta(calls=5):
    return {"calls_made": calls, "model": "test-model", "total_tokens": 500}


class TestCouncilQuery:
    @pytest.mark.asyncio
    async def test_missing_question(self):
        result = json.loads(await council_query(question=""))
        assert result["success"] is False
        assert "question" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_query(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())):
            result = json.loads(await council_query(question="Is X true?"))
            assert result["success"] is True
            assert result["confidence_score"] == 75
            assert "_meta" in result
            assert result["_meta"]["calls_made"] == 5
            assert "available_personas" in result

    @pytest.mark.asyncio
    async def test_all_fail(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(None, {"error": "All failed"})):
            result = json.loads(await council_query(question="Test?"))
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_persona_content_truncated(self):
        verdict = _mock_verdict()
        verdict.responses["advocate"] = PersonaResponse("advocate", "x" * 5000, 0.85, False, [], [])
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(verdict, _mock_meta())):
            result = json.loads(await council_query(question="Test?"))
            assert len(result["persona_responses"]["advocate"]["content"]) <= 2000


class TestCouncilEvaluate:
    @pytest.mark.asyncio
    async def test_missing_content(self):
        result = json.loads(await council_evaluate(content=""))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_successful_evaluate(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())):
            result = json.loads(await council_evaluate(content="Some analysis text"))
            assert result["success"] is True
            assert "criteria" in result
            assert "persona_feedback" in result


class TestCouncilGate:
    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = json.loads(await council_gate(action=""))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_allowed_above_threshold(self):
        with patch("hermes_council.server._run_gate", new_callable=AsyncMock, return_value=(_mock_verdict(confidence=70), _mock_meta(calls=3))):
            result = json.loads(await council_gate(action="Deploy", risk_level="medium"))
            assert result["allowed"] is True
            assert result["threshold"] == 50

    @pytest.mark.asyncio
    async def test_denied_below_threshold(self):
        with patch("hermes_council.server._run_gate", new_callable=AsyncMock, return_value=(_mock_verdict(confidence=40), _mock_meta(calls=3))):
            result = json.loads(await council_gate(action="Deploy", risk_level="high"))
            assert result["allowed"] is False
            assert result["threshold"] == 70

    @pytest.mark.asyncio
    async def test_skeptic_concerns_in_response(self):
        with patch("hermes_council.server._run_gate", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta(calls=3))):
            result = json.loads(await council_gate(action="Deploy"))
            assert "skeptic_concerns" in result
