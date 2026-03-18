"""FastMCP stdio server exposing council tools."""

import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from hermes_council.deliberation import _run_council, _run_gate
from hermes_council.personas import list_personas

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
    if not question.strip():
        return json.dumps({"success": False, "error": "question is required"})

    verdict, meta = await _run_council(
        question=question,
        context=context,
        persona_names=personas,
        evidence_search=evidence_search,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = {
        "success": True,
        "question": verdict.question,
        "confidence_score": verdict.confidence_score,
        "conflict_detected": verdict.conflict_detected,
        "arbiter_synthesis": verdict.arbiter_synthesis,
        "persona_responses": {
            name: {
                "confidence": resp.confidence,
                "dissents": resp.dissents,
                "key_points": resp.key_points,
                "content": resp.content[:2000],
                "sources": resp.sources,
            }
            for name, resp in verdict.responses.items()
        },
        "dpo_pairs": verdict.dpo_pairs,
        "sources": verdict.sources,
        "available_personas": list_personas(),
        "_meta": meta,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def council_evaluate(
    content: str,
    question: str = "",
    criteria: Optional[list[str]] = None,
) -> str:
    """Evaluate content quality through adversarial council critique. Returns confidence score and structured feedback."""
    if not content.strip():
        return json.dumps({"success": False, "error": "content is required"})

    if criteria is None:
        criteria = ["accuracy", "depth", "evidence", "falsifiability"]

    eval_question = f"Evaluate this content against: {', '.join(criteria)}.\n\n"
    if question:
        eval_question += f"Original task: {question}\n\n"
    eval_question += f"Content to evaluate:\n{content[:4000]}"

    verdict, meta = await _run_council(
        question=eval_question,
        context="This is an evaluation task. Each persona should critique the content from their intellectual tradition.",
        evidence_search=False,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = {
        "success": True,
        "confidence_score": verdict.confidence_score,
        "conflict_detected": verdict.conflict_detected,
        "criteria": criteria,
        "arbiter_synthesis": verdict.arbiter_synthesis,
        "persona_feedback": {
            name: {
                "confidence": resp.confidence,
                "dissents": resp.dissents,
                "key_points": resp.key_points,
                "content": resp.content[:1500],
            }
            for name, resp in verdict.responses.items()
        },
        "dpo_pairs": verdict.dpo_pairs,
        "sources": verdict.sources,
        "_meta": meta,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def council_gate(
    action: str,
    risk_level: str = "medium",
    context: str = "",
) -> str:
    """Quick safety review before high-stakes actions. Uses Skeptic + Oracle + Arbiter. Returns allow/deny with reasoning."""
    if not action.strip():
        return json.dumps({"success": False, "error": "action is required"})

    verdict, meta = await _run_gate(
        action=action,
        risk_level=risk_level,
        context=context,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    threshold = {"low": 30, "medium": 50, "high": 70}.get(risk_level, 50)
    allowed = verdict.confidence_score >= threshold

    skeptic_resp = verdict.responses.get("skeptic")
    skeptic_concerns = skeptic_resp.key_points if skeptic_resp else []

    result = {
        "success": True,
        "allowed": allowed,
        "confidence": verdict.confidence_score,
        "risk_level": risk_level,
        "threshold": threshold,
        "reasoning": verdict.arbiter_synthesis[:1000],
        "skeptic_concerns": skeptic_concerns,
        "_meta": meta,
    }
    return json.dumps(result, ensure_ascii=False)


def main():
    """Entry point for hermes-council-server."""
    mcp.run(transport="stdio")
