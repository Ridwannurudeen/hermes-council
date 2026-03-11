"""Tests for core deliberation logic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_council.deliberation import (
    _extract_dpo_pairs,
    _run_council,
    _run_gate,
    llm_call,
    _build_persona_response,
)
from hermes_council.personas import CouncilVerdict, PersonaResponse


def _mock_persona_json(confidence=0.8, dissent=False):
    return json.dumps({
        "reasoning": "Test analysis.",
        "confidence": confidence,
        "dissent": dissent,
        "key_points": ["point 1"],
        "sources": ["https://example.com"],
    })


def _mock_arbiter_json(confidence=0.75):
    return json.dumps({
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
    })


class TestLLMCall:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        with patch("hermes_council.deliberation.get_client", return_value=None):
            result, tokens = await llm_call("system", "user")
            assert "No API key" in result
            assert tokens == 0

    @pytest.mark.asyncio
    async def test_successful_call(self):
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


class TestBuildPersonaResponse:
    def test_valid_json(self):
        raw = _mock_persona_json(confidence=0.85, dissent=True)
        resp = _build_persona_response("skeptic", raw)
        assert resp.persona_name == "skeptic"
        assert resp.confidence == 0.85
        assert resp.dissents is True

    def test_valid_arbiter_json(self):
        raw = _mock_arbiter_json()
        resp = _build_persona_response("arbiter", raw, is_arbiter=True)
        assert resp.confidence == 0.75

    def test_invalid_json_fallback(self):
        raw = "Not JSON. CONFIDENCE: 0.6\nDISSENT: true\n- A key point here for testing"
        resp = _build_persona_response("oracle", raw)
        assert resp.confidence == 0.6
        assert resp.dissents is True


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
            assert meta["total_tokens"] == 600

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
            verdict, meta = await _run_council("Test?", persona_names=["skeptic", "arbiter"])
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
            assert len(verdict.responses) >= 3

    @pytest.mark.asyncio
    async def test_all_deliberators_fail(self):
        call_count = 0
        async def mock_llm(system, user, model=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise Exception("All fail")
            return _mock_arbiter_json(), 200

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
            "oracle": PersonaResponse("oracle", "Data", 0.8, False, [], []),
            "arbiter": PersonaResponse("arbiter", "Agreed", 0.85, False, [], []),
        }
        pairs = _extract_dpo_pairs("Question?", responses)
        assert len(pairs) == 0

    def test_empty_responses(self):
        pairs = _extract_dpo_pairs("Question?", {})
        assert pairs == []
