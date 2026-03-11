"""Regex fallback parsers for non-JSON-mode providers.

These are used when the LLM provider does not support
response_format={"type": "json_object"}. Ported from the original
hermes-agent PR #848 council_tool.py.
"""

import re
from typing import List

from hermes_council.personas import PersonaResponse


def parse_confidence(text: str) -> float:
    """Extract confidence value from persona response text."""
    patterns = [
        r"CONFIDENCE:\s*([\d.]+)",
        r"confidence[:\s]+([\d.]+)",
        r"(\d+)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
    return 0.5


def parse_dissent(text: str) -> bool:
    """Extract dissent flag from persona response text."""
    match = re.search(r"DISSENT:\s*(true|false)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return False


def parse_key_points(text: str) -> List[str]:
    """Extract bullet points from persona response text."""
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "  - ", "  * ")):
            point = line.lstrip("-* ").strip()
            if len(point) > 10:
                points.append(point)
    return points[:10]


def extract_sources(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
    return list(set(re.findall(url_pattern, text)))


def parse_persona_response(persona_name: str, raw_text: str) -> PersonaResponse:
    """Parse raw LLM output into a structured PersonaResponse using regex."""
    return PersonaResponse(
        persona_name=persona_name,
        content=raw_text,
        confidence=parse_confidence(raw_text),
        dissents=parse_dissent(raw_text),
        key_points=parse_key_points(raw_text),
        sources=extract_sources(raw_text),
    )
