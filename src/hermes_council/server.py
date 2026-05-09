"""FastMCP stdio server exposing council tools."""

import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from hermes_council.audit import write_audit_record
from hermes_council.deliberation import _run_council, _run_gate
from hermes_council.personas import CouncilVerdict, list_personas

# All logging to stderr — stdout is the MCP JSON-RPC protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hermes_council")

mcp = FastMCP("hermes-council")


def _arbiter_metadata(verdict: CouncilVerdict) -> dict:
    arbiter = verdict.responses.get("arbiter")
    return arbiter.metadata if arbiter else {}


def _action_summary(verdict: CouncilVerdict) -> dict:
    metadata = _arbiter_metadata(verdict)
    return {
        "recommendation": metadata.get("recommendation") or verdict.arbiter_synthesis[:300],
        "top_risks": metadata.get("top_risks") or [],
        "missing_evidence": metadata.get("missing_evidence") or [],
        "next_actions": metadata.get("next_actions") or [],
    }


def _persona_payload(verdict: CouncilVerdict, limit: int) -> dict:
    return {
        name: {
            "confidence": resp.confidence,
            "dissents": resp.dissents,
            "key_points": resp.key_points,
            "content": resp.content[:limit],
            "sources": resp.sources,
        }
        for name, resp in verdict.responses.items()
    }


def _record_audit(tool: str, request: dict, result: dict) -> None:
    audit_path = write_audit_record(tool, request, result)
    if audit_path:
        result.setdefault("_meta", {})["audit_path"] = audit_path


def _base_result(verdict: CouncilVerdict, meta: dict) -> dict:
    return {
        "success": True,
        "question": verdict.question,
        "confidence_score": verdict.confidence_score,
        "conflict_detected": verdict.conflict_detected,
        "arbiter_synthesis": verdict.arbiter_synthesis,
        "action_summary": _action_summary(verdict),
        "dpo_pairs": verdict.dpo_pairs,
        "sources": verdict.sources,
        "verified_sources": verdict.verified_sources,
        "_meta": meta,
    }


def _normalize_gate_verdict(raw: str) -> str:
    normalized = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"allow", "allow_with_conditions", "deny"}:
        return normalized
    return "not_applicable"


def _gate_result(verdict: CouncilVerdict, meta: dict, risk_level: str) -> dict:
    metadata = _arbiter_metadata(verdict)
    threshold = {"low": 30, "medium": 50, "high": 70, "critical": 85}.get(risk_level, 50)
    decision = _normalize_gate_verdict(metadata.get("verdict", ""))

    skeptic_resp = verdict.responses.get("skeptic")
    skeptic_concerns = skeptic_resp.key_points if skeptic_resp else []
    blocking_risks = metadata.get("blocking_risks") or []
    required_checks = metadata.get("required_checks") or []

    if decision == "not_applicable":
        if verdict.confidence_score < threshold:
            decision = "deny"
            blocking_risks = blocking_risks or skeptic_concerns or metadata.get("top_risks") or []
        elif skeptic_concerns and risk_level in {"high", "critical"}:
            decision = "allow_with_conditions"
            required_checks = required_checks or skeptic_concerns
        else:
            decision = "allow"

    result = {
        "success": True,
        "verdict": decision,
        "allowed": decision != "deny",
        "can_proceed_now": decision == "allow",
        "requires_conditions": decision == "allow_with_conditions",
        "confidence": verdict.confidence_score,
        "confidence_in_verdict": verdict.confidence_score,
        "risk_level": risk_level,
        "threshold": threshold,
        "reasoning": verdict.arbiter_synthesis[:1000],
        "blocking_risks": blocking_risks,
        "required_checks": required_checks,
        "safe_alternative": metadata.get("safe_alternative", ""),
        "skeptic_concerns": skeptic_concerns,
        "action_summary": _action_summary(verdict),
        "_meta": meta,
    }
    return result


async def _run_review(
    *,
    tool_name: str,
    request: dict,
    question: str,
    context: str = "",
    evidence_search: bool = False,
    mode: str = "standard",
) -> str:
    verdict, meta = await _run_council(
        question=question,
        context=context,
        evidence_search=evidence_search,
        mode=mode,
    )
    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = _base_result(verdict, meta)
    result["persona_responses"] = _persona_payload(verdict, 1500)
    _record_audit(tool_name, request, result)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def council_query(
    question: str,
    context: str = "",
    personas: Optional[list[str]] = None,
    evidence_search: bool = True,
    mode: str = "standard",
    max_tokens: int = 2000,
    max_evidence_sources: int = 5,
) -> str:
    """Submit a question for 5-persona adversarial deliberation. Returns structured verdict with confidence score, evidence links, and DPO pairs."""
    if not question.strip():
        return json.dumps({"success": False, "error": "question is required"})

    verdict, meta = await _run_council(
        question=question,
        context=context,
        persona_names=personas,
        evidence_search=evidence_search,
        mode=mode,
        max_tokens=max_tokens,
        max_evidence_sources=max_evidence_sources,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = _base_result(verdict, meta)
    result["persona_responses"] = _persona_payload(verdict, 2000)
    result["available_personas"] = list_personas()
    _record_audit(
        "council_query",
        {
            "question": question,
            "context": context,
            "personas": personas,
            "evidence_search": evidence_search,
            "mode": mode,
        },
        result,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def council_evaluate(
    content: str,
    question: str = "",
    criteria: Optional[list[str]] = None,
    mode: str = "standard",
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
        mode=mode,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = _base_result(verdict, meta)
    result["criteria"] = criteria
    result["persona_feedback"] = _persona_payload(verdict, 1500)
    _record_audit(
        "council_evaluate",
        {"content": content[:4000], "question": question, "criteria": criteria, "mode": mode},
        result,
    )
    return json.dumps(result, ensure_ascii=False)


async def _gate_response(
    *,
    tool_name: str,
    request: dict,
    action: str,
    risk_level: str,
    context: str,
) -> str:
    if not action.strip():
        return json.dumps({"success": False, "error": "action is required"})

    verdict, meta = await _run_gate(
        action=action,
        risk_level=risk_level,
        context=context,
    )

    if verdict is None:
        return json.dumps({"success": False, **meta})

    result = _gate_result(verdict, meta, risk_level)
    _record_audit(tool_name, request, result)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def council_gate(
    action: str,
    risk_level: str = "medium",
    context: str = "",
) -> str:
    """Quick safety review before high-stakes actions. Returns allow/conditional/deny with reasoning."""
    return await _gate_response(
        tool_name="council_gate",
        request={"action": action, "risk_level": risk_level, "context": context},
        action=action,
        risk_level=risk_level,
        context=context,
    )


@mcp.tool()
async def council_preflight(
    action: str,
    risk_level: str = "medium",
    context: str = "",
    checks: Optional[list[str]] = None,
) -> str:
    """Preflight a risky action and return a structured allow/conditional/deny verdict."""
    effective_context = context
    if checks:
        effective_context = (
            f"{context}\nRequired checks already considered: {', '.join(checks)}".strip()
        )
    return await _gate_response(
        tool_name="council_preflight",
        request={
            "action": action,
            "risk_level": risk_level,
            "context": context,
            "checks": checks,
        },
        action=action,
        risk_level=risk_level,
        context=effective_context,
    )


@mcp.tool()
async def council_review_plan(
    plan: str,
    objective: str = "",
    risk_level: str = "medium",
    mode: str = "standard",
) -> str:
    """Review an implementation or execution plan for gaps, risks, and next actions."""
    if not plan.strip():
        return json.dumps({"success": False, "error": "plan is required"})

    question = (
        "Review this plan before execution. Identify blocking risks, missing steps, "
        "weak assumptions, and the strongest recommended next action.\n\n"
        f"Risk level: {risk_level}\n"
    )
    if objective:
        question += f"Objective: {objective}\n"
    question += f"\nPlan:\n{plan[:6000]}"
    return await _run_review(
        tool_name="council_review_plan",
        request={"plan": plan[:6000], "objective": objective, "risk_level": risk_level, "mode": mode},
        question=question,
        evidence_search=False,
        mode=mode,
    )


@mcp.tool()
async def council_review_diff(
    diff: str,
    objective: str = "",
    files: Optional[list[str]] = None,
    risk_level: str = "high",
    mode: str = "standard",
) -> str:
    """Review a code diff for defects, regressions, missing tests, and risky changes."""
    if not diff.strip():
        return json.dumps({"success": False, "error": "diff is required"})

    question = (
        "Review this code diff as a production reviewer. Prioritize bugs, security "
        "risks, behavioral regressions, missing tests, and integration failures.\n\n"
        f"Risk level: {risk_level}\n"
    )
    if objective:
        question += f"Objective: {objective}\n"
    if files:
        question += f"Files: {', '.join(files)}\n"
    question += f"\nDiff:\n{diff[:8000]}"
    return await _run_review(
        tool_name="council_review_diff",
        request={
            "diff": diff[:8000],
            "objective": objective,
            "files": files,
            "risk_level": risk_level,
            "mode": mode,
        },
        question=question,
        evidence_search=False,
        mode=mode,
    )


@mcp.tool()
async def council_review_claim(
    claim: str,
    context: str = "",
    evidence_search: bool = True,
    mode: str = "standard",
) -> str:
    """Fact-check or stress-test a claim with evidence-grounded council review."""
    if not claim.strip():
        return json.dumps({"success": False, "error": "claim is required"})

    question = (
        "Evaluate this claim. Separate verified evidence from speculation, identify "
        "what would falsify it, and give a calibrated recommendation.\n\n"
        f"Claim: {claim}"
    )
    return await _run_review(
        tool_name="council_review_claim",
        request={"claim": claim, "context": context, "evidence_search": evidence_search, "mode": mode},
        question=question,
        context=context,
        evidence_search=evidence_search,
        mode=mode,
    )


@mcp.tool()
async def council_decision(
    options: list[str],
    decision_context: str = "",
    criteria: Optional[list[str]] = None,
    mode: str = "standard",
) -> str:
    """Compare options and return a recommended decision with risks and next actions."""
    options = [option for option in options if option.strip()]
    if len(options) < 2:
        return json.dumps({"success": False, "error": "at least two options are required"})

    criteria = criteria or ["impact", "risk", "reversibility", "evidence"]
    option_lines = "\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1))
    question = (
        "Choose the strongest option. Recommend exactly one option unless the evidence "
        "is insufficient, then state what evidence is missing.\n\n"
        f"Criteria: {', '.join(criteria)}\n"
    )
    if decision_context:
        question += f"Context: {decision_context}\n"
    question += f"\nOptions:\n{option_lines}"
    return await _run_review(
        tool_name="council_decision",
        request={
            "options": options,
            "decision_context": decision_context,
            "criteria": criteria,
            "mode": mode,
        },
        question=question,
        evidence_search=False,
        mode=mode,
    )


def main():
    """Entry point for hermes-council-server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
