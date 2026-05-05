"""Tests for FastMCP server tool handlers."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from hermes_council.personas import CouncilVerdict, PersonaResponse
from hermes_council.server import (
    council_decision,
    council_evaluate,
    council_gate,
    council_preflight,
    council_query,
    council_review_claim,
    council_review_diff,
    council_review_plan,
    main,
)


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


def _mock_gate_verdict(decision="allow_with_conditions"):
    verdict = _mock_verdict(confidence=80)
    verdict.responses["arbiter"].metadata = {
        "verdict": decision,
        "recommendation": "Proceed only after checks.",
        "top_risks": ["migration risk"],
        "missing_evidence": ["rollback proof"],
        "next_actions": ["run migration dry-run"],
        "blocking_risks": ["no backup"],
        "required_checks": ["backup verified"],
        "safe_alternative": "stage the deploy",
    }
    return verdict


def _mock_meta(calls=5):
    return {"calls_made": calls, "model": "test-model", "total_tokens": 500, "mode": "standard"}


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
            assert "action_summary" in result
            assert "verified_sources" in result

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

    @pytest.mark.asyncio
    async def test_metadata_verdict_can_require_conditions(self):
        with patch("hermes_council.server._run_gate", new_callable=AsyncMock, return_value=(_mock_gate_verdict(), _mock_meta(calls=3))):
            result = json.loads(await council_gate(action="Deploy", risk_level="high"))
            assert result["verdict"] == "allow_with_conditions"
            assert result["allowed"] is True
            assert result["can_proceed_now"] is False
            assert result["required_checks"] == ["backup verified"]
            assert result["safe_alternative"] == "stage the deploy"


class TestReviewTools:
    @pytest.mark.asyncio
    async def test_review_plan_requires_plan(self):
        result = json.loads(await council_review_plan(plan=""))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_review_plan_success(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())) as run:
            result = json.loads(await council_review_plan(plan="1. Change auth", objective="Reduce risk"))
            assert result["success"] is True
            assert "persona_responses" in result
            assert "action_summary" in result
            assert "Change auth" in run.call_args.kwargs["question"]

    @pytest.mark.asyncio
    async def test_review_diff_success(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())) as run:
            result = json.loads(await council_review_diff(diff="+ risky change", files=["src/app.py"]))
            assert result["success"] is True
            assert "src/app.py" in run.call_args.kwargs["question"]

    @pytest.mark.asyncio
    async def test_review_claim_uses_evidence_by_default(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())) as run:
            result = json.loads(await council_review_claim(claim="Python supports async functions."))
            assert result["success"] is True
            assert run.call_args.kwargs["evidence_search"] is True

    @pytest.mark.asyncio
    async def test_decision_requires_two_options(self):
        result = json.loads(await council_decision(options=["Ship"]))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_decision_success(self):
        with patch("hermes_council.server._run_council", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta())) as run:
            result = json.loads(await council_decision(options=["Ship", "Delay"], criteria=["risk"]))
            assert result["success"] is True
            assert "Ship" in run.call_args.kwargs["question"]
            assert "Delay" in run.call_args.kwargs["question"]

    @pytest.mark.asyncio
    async def test_preflight_forwards_checks(self):
        with patch("hermes_council.server._run_gate", new_callable=AsyncMock, return_value=(_mock_verdict(), _mock_meta(calls=3))) as run:
            result = json.loads(await council_preflight(action="Deploy", checks=["tests pass"]))
            assert result["success"] is True
            assert "tests pass" in run.call_args.kwargs["context"]


def test_main_runs_stdio_transport(monkeypatch):
    called = {}

    def fake_run(*, transport):
        called["transport"] = transport

    monkeypatch.setattr("hermes_council.server.mcp.run", fake_run)
    main()

    assert called["transport"] == "stdio"
