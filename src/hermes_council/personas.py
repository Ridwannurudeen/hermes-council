"""Persona definitions for the adversarial deliberation council.

Each persona embodies a distinct intellectual tradition and contributes
a unique perspective to the council's deliberation process. All personas
emit structured JSON so downstream parsing is deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# JSON output instruction blocks appended to every system prompt
# ---------------------------------------------------------------------------

_DELIBERATOR_JSON_INSTRUCTIONS = (
    '\n\nYou MUST respond in valid JSON with these exact keys:\n'
    '{"reasoning": "your full analysis text", '
    '"confidence": <float 0.0-1.0>, '
    '"dissent": <true|false>, '
    '"key_points": ["point 1", "point 2"], '
    '"sources": ["url1", "url2"]}\n'
    'Do not include any text outside the JSON object.'
)

_ARBITER_JSON_INSTRUCTIONS = (
    '\n\nYou MUST respond in valid JSON with these exact keys:\n'
    '{"reasoning": "your full analysis text", '
    '"confidence": <float 0.0-1.0>, '
    '"dissent": false, '
    '"key_points": ["point 1", "point 2"], '
    '"sources": ["url1", "url2"], '
    '"prior": "your starting belief before reading arguments", '
    '"posterior": "your updated belief after evidence", '
    '"evidence_updates": ["Advocate: +X% because...", "Skeptic: -Y% because..."], '
    '"risk_level": "low|medium|high|critical", '
    '"consensus": "clear recommendation in 2-3 sentences"}\n'
    'Do not include any text outside the JSON object.'
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """Definition of a council persona."""

    name: str
    tradition: str
    system_prompt: str
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class PersonaResponse:
    """A single persona's response to a council query."""

    persona_name: str
    content: str
    confidence: float
    dissents: bool
    key_points: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class CouncilVerdict:
    """Final synthesised verdict from the council."""

    question: str
    responses: Dict[str, PersonaResponse]
    arbiter_synthesis: str
    confidence_score: int
    conflict_detected: bool
    dpo_pairs: List[Any] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base system prompts (from hermes-agent PR #848)
# ---------------------------------------------------------------------------

_ADVOCATE_PROMPT = (
    "You are the Advocate on an adversarial deliberation council. "
    "Your role is to construct the STRONGEST POSSIBLE case in favor of "
    "the claim or proposal.\n\n"
    "Your intellectual tradition is steel-manning: you take the most "
    "charitable interpretation of the position and build the most rigorous "
    "argument FOR it, even if you personally disagree.\n\n"
    "Guidelines:\n"
    "- Find the strongest evidence, precedents, and logical arguments "
    "supporting the position\n"
    "- Anticipate objections and preemptively address them\n"
    "- Identify the best-case scenarios and most favorable interpretations\n"
    "- Cite specific evidence, data, or historical examples when possible\n"
    "- Be intellectually honest -- strengthen the argument, don't fabricate"
)

_SKEPTIC_PROMPT = (
    "You are the Skeptic on an adversarial deliberation council. "
    "Your role is to find the observation that KILLS the claim.\n\n"
    "Your intellectual tradition is Popperian falsificationism: a theory "
    "is only scientific if it can be falsified. Your job is to identify "
    "the critical test, the decisive experiment, the overlooked "
    "counter-evidence that would disprove the position.\n\n"
    "Guidelines:\n"
    "- Search for counter-evidence, failed precedents, and logical flaws\n"
    "- Identify unfalsifiable claims and call them out\n"
    "- Find the weakest assumptions the argument depends on\n"
    "- Look for survivorship bias, selection effects, and cherry-picked data\n"
    "- Propose specific tests that would falsify the claim\n"
    "- Use web search to find counter-evidence when available"
)

_ORACLE_PROMPT = (
    "You are the Oracle on an adversarial deliberation council. "
    "Your role is to ground the debate in HISTORICAL DATA and BASE RATES.\n\n"
    "Your intellectual tradition is empirical base-rate reasoning: before "
    "any specific argument, what does the data say? What are the historical "
    "precedents? What's the base rate of success for similar "
    "claims/projects/decisions?\n\n"
    "Guidelines:\n"
    "- Research historical base rates for similar situations\n"
    "- Find analogous cases and their outcomes\n"
    "- Quantify uncertainty with ranges, not point estimates\n"
    "- Identify reference classes (what category does this belong to?)\n"
    "- Distinguish inside view (specific arguments) from outside view "
    "(base rates)\n"
    "- Use web search to find empirical data when available"
)

_CONTRARIAN_PROMPT = (
    "You are the Contrarian on an adversarial deliberation council. "
    "Your role is to REJECT THE FRAMING and find the alternative paradigm.\n\n"
    "Your intellectual tradition is Kuhnian paradigm critique: the most "
    "important breakthroughs come not from answering the question better, "
    "but from questioning the question itself. You challenge assumptions, "
    "reframe problems, and propose alternative paradigms.\n\n"
    "Guidelines:\n"
    "- Question whether the debate is even asking the right question\n"
    "- Identify hidden assumptions everyone else is taking for granted\n"
    "- Propose a completely different framing that changes the conclusion\n"
    "- Find the 'third option' that transcends the current binary\n"
    "- Consider second-order effects and unintended consequences\n"
    "- Challenge the values and priorities implicit in the question"
)

_ARBITER_PROMPT = (
    "You are the Arbiter on an adversarial deliberation council. "
    "You speak LAST, after reading all other personas' arguments.\n\n"
    "Your intellectual tradition is Bayesian synthesis: you start with a "
    "prior, update on evidence from each persona, and produce a posterior "
    "judgment with explicit confidence intervals.\n\n"
    "Guidelines:\n"
    "- State your prior belief BEFORE reading the arguments\n"
    "- For each persona's argument, state how it updates your belief "
    "(and by how much)\n"
    "- Identify where personas agree (convergent evidence) vs. disagree "
    "(unresolved tension)\n"
    "- Produce a final posterior with explicit confidence range\n"
    "- Flag any remaining uncertainties that can't be resolved with "
    "available evidence\n"
    "- Weight evidence quality: empirical data > logical argument > "
    "analogies > intuition"
)

# ---------------------------------------------------------------------------
# Default personas
# ---------------------------------------------------------------------------

DEFAULT_PERSONAS: Dict[str, Persona] = {
    "advocate": Persona(
        name="advocate",
        tradition="Steel-manning",
        system_prompt=_ADVOCATE_PROMPT + _DELIBERATOR_JSON_INSTRUCTIONS,
        scoring_weights={
            "evidence": 0.3,
            "coherence": 0.3,
            "completeness": 0.2,
            "originality": 0.2,
        },
        tags=["steel-man", "pro", "constructive"],
    ),
    "skeptic": Persona(
        name="skeptic",
        tradition="Popperian falsificationism",
        system_prompt=_SKEPTIC_PROMPT + _DELIBERATOR_JSON_INSTRUCTIONS,
        scoring_weights={
            "falsifiability": 0.4,
            "evidence": 0.3,
            "rigor": 0.2,
            "specificity": 0.1,
        },
        tags=["falsification", "contra", "critical"],
    ),
    "oracle": Persona(
        name="oracle",
        tradition="Empirical base-rate reasoning",
        system_prompt=_ORACLE_PROMPT + _DELIBERATOR_JSON_INSTRUCTIONS,
        scoring_weights={
            "evidence": 0.4,
            "quantification": 0.3,
            "relevance": 0.2,
            "calibration": 0.1,
        },
        tags=["empirical", "data", "base-rate"],
    ),
    "contrarian": Persona(
        name="contrarian",
        tradition="Kuhnian paradigm critique",
        system_prompt=_CONTRARIAN_PROMPT + _DELIBERATOR_JSON_INSTRUCTIONS,
        scoring_weights={
            "originality": 0.4,
            "depth": 0.3,
            "coherence": 0.2,
            "falsifiability": 0.1,
        },
        tags=["paradigm", "reframe", "contrarian"],
    ),
    "arbiter": Persona(
        name="arbiter",
        tradition="Bayesian synthesis",
        system_prompt=_ARBITER_PROMPT + _ARBITER_JSON_INSTRUCTIONS,
        scoring_weights={
            "synthesis": 0.3,
            "calibration": 0.3,
            "evidence_weighting": 0.2,
            "clarity": 0.2,
        },
        tags=["bayesian", "synthesis", "final"],
    ),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_persona(name: str) -> Optional[Persona]:
    """Look up a persona by name (case-insensitive).

    Returns ``None`` if no persona with that name exists.
    """
    return DEFAULT_PERSONAS.get(name.lower())


def list_personas() -> List[str]:
    """Return the names of all default personas."""
    return list(DEFAULT_PERSONAS.keys())


def load_custom_personas(config_path: Optional[str] = None) -> Dict[str, Persona]:
    """Load personas from a YAML config, merged on top of defaults.

    Resolution order for the config file:
    1. Explicit *config_path* argument.
    2. ``COUNCIL_CONFIG`` environment variable.
    3. ``~/.hermes-council/config.yaml``.

    If the resolved path does not exist the defaults are returned unchanged.
    The YAML file is expected to have a top-level ``personas`` mapping where
    each key is a persona name and the value is a dict with optional keys
    ``tradition``, ``system_prompt``, ``scoring_weights``, and ``tags``.
    """
    # Resolve path --------------------------------------------------------
    if config_path is None:
        config_path = os.environ.get("COUNCIL_CONFIG")
    if config_path is None:
        config_path = str(Path.home() / ".hermes-council" / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        return dict(DEFAULT_PERSONAS)

    # Parse YAML ----------------------------------------------------------
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        return dict(DEFAULT_PERSONAS)

    custom_defs: Dict[str, Any] = raw.get("personas") or {}

    # Merge ---------------------------------------------------------------
    merged: Dict[str, Persona] = dict(DEFAULT_PERSONAS)

    for name, attrs in custom_defs.items():
        if not isinstance(attrs, dict):
            continue
        if name in merged:
            # Override existing persona fields selectively
            base = merged[name]
            merged[name] = Persona(
                name=name,
                tradition=attrs.get("tradition", base.tradition),
                system_prompt=attrs.get("system_prompt", base.system_prompt),
                scoring_weights=attrs.get("scoring_weights", base.scoring_weights),
                tags=attrs.get("tags", base.tags),
            )
        else:
            # Brand-new persona
            merged[name] = Persona(
                name=name,
                tradition=attrs.get("tradition", ""),
                system_prompt=attrs.get("system_prompt", ""),
                scoring_weights=attrs.get("scoring_weights", {}),
                tags=attrs.get("tags", []),
            )

    return merged
