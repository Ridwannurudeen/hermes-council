"""Core deliberation engine for the adversarial council.

Orchestrates parallel LLM calls across personas, handles JSON mode
negotiation with fallback to regex parsing, detects conflict, runs the
Arbiter synthesis, and extracts DPO preference pairs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple

from hermes_council.client import (
    get_client,
    get_model,
    is_json_mode_supported,
    set_json_mode_supported,
)
from hermes_council.evidence import EvidenceBundle, collect_evidence
from hermes_council.parsing import parse_persona_response
from hermes_council.personas import (
    CouncilVerdict,
    PersonaResponse,
    load_custom_personas,
)
from hermes_council.schemas import ArbiterOutput, PersonaOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM call with JSON mode negotiation
# ---------------------------------------------------------------------------


async def llm_call(
    system_prompt: str,
    user_message: str,
    model: str = None,
    max_tokens: int = 2000,
) -> Tuple[str, int]:
    """Make a single LLM call via AsyncOpenAI.

    Returns (content_string, token_count).  Automatically negotiates
    JSON mode support: tries ``response_format={"type": "json_object"}``
    on the first call and falls back if the provider rejects it.
    """
    client = get_client()
    if client is None:
        return ("No API key configured. Set COUNCIL_API_KEY, OPENROUTER_API_KEY, "
                "NOUS_API_KEY, or OPENAI_API_KEY."), 0

    model = model or get_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    base_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    json_supported = is_json_mode_supported()

    # Case 1: already known to NOT support JSON mode
    if json_supported is False:
        response = await client.chat.completions.create(**base_kwargs)
        content = response.choices[0].message.content or ""
        tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
        return content, tokens

    # Case 2: supported (True) or untested (None) -- try with JSON mode
    try:
        kwargs = {**base_kwargs, "response_format": {"type": "json_object"}}
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0

        # If we got here and it was untested, mark as supported
        if json_supported is None:
            set_json_mode_supported(True)

        return content, tokens

    except Exception as exc:
        # Check if it's a BadRequestError about response_format
        from openai import BadRequestError
        if isinstance(exc, BadRequestError):
            logger.warning("JSON mode rejected by provider, retrying without it: %s", exc)
            set_json_mode_supported(False)
            response = await client.chat.completions.create(**base_kwargs)
            content = response.choices[0].message.content or ""
            tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
            return content, tokens
        raise


# ---------------------------------------------------------------------------
# Build PersonaResponse from raw LLM text
# ---------------------------------------------------------------------------


def _build_persona_response(
    persona_name: str,
    raw_text: str,
    is_arbiter: bool = False,
) -> PersonaResponse:
    """Parse raw LLM output into a PersonaResponse.

    Tries JSON parsing first (using the appropriate Pydantic model),
    then falls back to regex-based extraction.
    """
    try:
        data = json.loads(raw_text)
        if is_arbiter:
            parsed = ArbiterOutput.model_validate(data)
            metadata = {
                "prior": parsed.prior,
                "posterior": parsed.posterior,
                "evidence_updates": parsed.evidence_updates,
                "risk_level": parsed.risk_level,
                "consensus": parsed.consensus,
                "recommendation": parsed.recommendation,
                "top_risks": parsed.top_risks,
                "missing_evidence": parsed.missing_evidence,
                "next_actions": parsed.next_actions,
                "verdict": parsed.verdict,
                "blocking_risks": parsed.blocking_risks,
                "required_checks": parsed.required_checks,
                "safe_alternative": parsed.safe_alternative,
            }
        else:
            parsed = PersonaOutput.model_validate(data)
            metadata = {}

        return PersonaResponse(
            persona_name=persona_name,
            content=parsed.reasoning,
            confidence=parsed.confidence,
            dissents=parsed.dissent,
            key_points=parsed.key_points,
            sources=parsed.sources,
            metadata=metadata,
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(
            "JSON parse failed for %s, falling back to regex: %s",
            persona_name, exc,
        )
        return parse_persona_response(persona_name, raw_text)


# ---------------------------------------------------------------------------
# Council orchestration
# ---------------------------------------------------------------------------


def _mode_personas(mode: str) -> Optional[List[str]]:
    """Return persona order for a named council mode."""
    if mode == "fast":
        return ["skeptic", "arbiter"]
    if mode in {"standard", "deep"}:
        return None
    return None


async def _run_council(
    question: str,
    context: str = "",
    persona_names: Optional[List[str]] = None,
    evidence_search: bool = True,
    model: Optional[str] = None,
    mode: str = "standard",
    max_tokens: int = 2000,
    max_evidence_sources: int = 5,
) -> Tuple[Optional[CouncilVerdict], dict]:
    """Run a full council deliberation.

    1. Loads personas (custom config merged over defaults).
    2. Runs all deliberator personas in parallel.
    3. Detects conflict (confidence spread > 0.3).
    4. Runs the Arbiter with all deliberator arguments.
    5. Aggregates sources, extracts DPO pairs, returns verdict + meta.
    """
    model_used = model or get_model()
    mode = (mode or "standard").lower()
    if get_client() is None:
        return None, {
            "error": (
                "No API key configured. Set COUNCIL_API_KEY, OPENROUTER_API_KEY, "
                "NOUS_API_KEY, or OPENAI_API_KEY."
            ),
            "calls_made": 0,
            "model": model_used,
            "total_tokens": 0,
            "mode": mode,
        }

    # Load and partition personas
    all_personas = load_custom_personas()
    personas = dict(all_personas)  # copy before mutating

    if persona_names is None:
        persona_names = _mode_personas(mode)

    # If specific personas requested, filter to those
    if persona_names is not None:
        personas = {
            name: personas[name]
            for name in persona_names
            if name in personas
        }

    # Pop the arbiter out of the deliberators
    arbiter = personas.pop("arbiter", all_personas.get("arbiter"))

    # Build user message
    user_msg = question
    if context:
        user_msg += f"\n\nContext: {context}"

    evidence_bundle = EvidenceBundle([], [])
    if evidence_search:
        evidence_bundle = collect_evidence(
            question,
            context,
            enabled=True,
            max_sources=max_evidence_sources,
        )
        user_msg += f"\n\n{evidence_bundle.to_prompt_block()}"

    # Run deliberators in parallel
    total_tokens = 0
    calls_made = 0

    async def _call_persona(name, persona):
        return name, await llm_call(
            persona.system_prompt,
            user_msg,
            model=model_used,
            max_tokens=max_tokens,
        )

    tasks = [
        _call_persona(name, persona)
        for name, persona in personas.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid responses, track failures
    deliberator_responses: Dict[str, PersonaResponse] = {}
    exceptions = []

    for result in results:
        if isinstance(result, Exception):
            exceptions.append(result)
            logger.error("Deliberator failed: %s", result)
            continue

        name, (raw_text, tokens) = result
        total_tokens += tokens
        calls_made += 1
        deliberator_responses[name] = _build_persona_response(name, raw_text)

    # All deliberators failed
    if not deliberator_responses:
        return None, {
            "error": "All council deliberators failed",
            "details": [str(e) for e in exceptions],
        }

    # Detect conflict: confidence spread > 0.3
    confidences = [r.confidence for r in deliberator_responses.values()]
    confidence_spread = max(confidences) - min(confidences)
    conflict_detected = confidence_spread > 0.3

    # Build Arbiter context
    arbiter_context_parts = [
        f"Question: {question}\n\n"
        "Here are the arguments from each council member:\n"
    ]
    for name, resp in deliberator_responses.items():
        truncated_content = resp.content[:3000]
        arbiter_context_parts.append(
            f"\n--- {name.upper()} (confidence: {resp.confidence}) ---\n"
            f"{truncated_content}\n"
        )

    if conflict_detected:
        arbiter_context_parts.append(
            "\n[CONFLICT DETECTED: Significant disagreement among council members. "
            "Pay special attention to resolving the tension.]\n"
        )

    arbiter_user_msg = "".join(arbiter_context_parts)

    # Run Arbiter
    arbiter_raw, arbiter_tokens = await llm_call(
        arbiter.system_prompt,
        arbiter_user_msg,
        model=model_used,
        max_tokens=max_tokens,
    )
    total_tokens += arbiter_tokens
    calls_made += 1

    arbiter_response = _build_persona_response("arbiter", arbiter_raw, is_arbiter=True)

    if mode == "deep":
        second_pass_msg = (
            f"Question: {question}\n\n"
            "Review your previous synthesis for overconfidence, missing evidence, "
            "and unresolved dissent. Produce the final JSON verdict.\n\n"
            f"Previous synthesis:\n{arbiter_response.content[:3000]}\n"
        )
        if evidence_bundle.sources:
            second_pass_msg += f"\n\n{evidence_bundle.to_prompt_block()}"
        arbiter_raw, arbiter_tokens = await llm_call(
            arbiter.system_prompt,
            second_pass_msg,
            model=model_used,
            max_tokens=max_tokens,
        )
        total_tokens += arbiter_tokens
        calls_made += 1
        arbiter_response = _build_persona_response("arbiter", arbiter_raw, is_arbiter=True)

    # Aggregate all responses (deliberators + arbiter)
    all_responses = {**deliberator_responses, "arbiter": arbiter_response}

    # Aggregate sources
    all_sources = []
    for resp in all_responses.values():
        all_sources.extend(resp.sources)
    unique_sources = list(dict.fromkeys(all_sources))  # dedup preserving order

    # Confidence score
    confidence_score = int(arbiter_response.confidence * 100)

    # Extract DPO pairs
    dpo_pairs = _extract_dpo_pairs(question, all_responses)

    verdict = CouncilVerdict(
        question=question,
        responses=all_responses,
        arbiter_synthesis=arbiter_response.content,
        confidence_score=confidence_score,
        conflict_detected=conflict_detected,
        dpo_pairs=dpo_pairs,
        sources=unique_sources,
        verified_sources=[source.to_dict() for source in evidence_bundle.verified_sources],
    )

    meta = {
        "calls_made": calls_made,
        "model": model_used,
        "total_tokens": total_tokens,
        "mode": mode,
        "max_tokens": max_tokens,
        "evidence_enabled": evidence_search,
        "evidence_sources": [source.to_dict() for source in evidence_bundle.sources],
        "verified_source_count": len(evidence_bundle.verified_sources),
        "evidence_errors": evidence_bundle.errors,
    }

    return verdict, meta


# ---------------------------------------------------------------------------
# DPO pair extraction
# ---------------------------------------------------------------------------


def _extract_dpo_pairs(
    question: str,
    responses: Dict[str, PersonaResponse],
) -> List[dict]:
    """Extract DPO preference pairs from council responses.

    Pair 1: Arbiter (chosen) vs lowest-confidence dissenter (rejected).
    Pair 2: Highest-confidence aligned vs lowest-confidence persona
            (if confidence spread > 0.2).
    """
    if not responses:
        return []

    pairs: List[dict] = []

    arbiter = responses.get("arbiter")
    non_arbiter = {k: v for k, v in responses.items() if k != "arbiter"}

    if not non_arbiter:
        return []

    # Find dissenters
    dissenters = {k: v for k, v in non_arbiter.items() if v.dissents}

    # Pair 1: Arbiter vs lowest-confidence dissenter
    if arbiter and dissenters:
        lowest_dissenter_name = min(dissenters, key=lambda k: dissenters[k].confidence)
        lowest_dissenter = dissenters[lowest_dissenter_name]
        pairs.append({
            "question": question,
            "chosen": arbiter.content,
            "rejected": lowest_dissenter.content,
            "confidence": arbiter.confidence,
            "source": "council_evaluation",
            "chosen_persona": "arbiter",
            "rejected_persona": lowest_dissenter_name,
        })

    # Pair 2: Highest-confidence aligned vs lowest-confidence (if spread > 0.2)
    all_confidences = [v.confidence for v in non_arbiter.values()]
    if len(all_confidences) >= 2:
        spread = max(all_confidences) - min(all_confidences)
        if spread > 0.2:
            aligned = {k: v for k, v in non_arbiter.items() if not v.dissents}
            if aligned:
                highest_aligned_name = max(aligned, key=lambda k: aligned[k].confidence)
                highest_aligned = aligned[highest_aligned_name]
                lowest_name = min(non_arbiter, key=lambda k: non_arbiter[k].confidence)
                lowest = non_arbiter[lowest_name]
                # Only add if they're different personas
                if highest_aligned_name != lowest_name:
                    pairs.append({
                        "question": question,
                        "chosen": highest_aligned.content,
                        "rejected": lowest.content,
                        "confidence": highest_aligned.confidence,
                        "source": "council_evaluation",
                        "chosen_persona": highest_aligned_name,
                        "rejected_persona": lowest_name,
                    })

    return pairs


# ---------------------------------------------------------------------------
# Gate mode (lightweight 3-persona check)
# ---------------------------------------------------------------------------


async def _run_gate(
    action: str,
    risk_level: str = "medium",
    context: str = "",
) -> Tuple[Optional[CouncilVerdict], dict]:
    """Run a lightweight gate check using skeptic + oracle + arbiter.

    Used for quick go/no-go decisions before executing an action.
    """
    gate_question = (
        "Preflight this action. The Arbiter must set verdict to one of "
        "allow, allow_with_conditions, or deny. Use deny for unresolved "
        "blocking risks. Use allow_with_conditions when specific checks are "
        "required before proceeding.\n\n"
        f"Action: {action}\n"
        f"Risk level: {risk_level}"
    )
    if context:
        gate_question += f"\nContext: {context}"

    return await _run_council(
        gate_question,
        persona_names=["skeptic", "oracle", "arbiter"],
        evidence_search=False,
        mode="fast",
    )
