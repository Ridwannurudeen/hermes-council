"""Stub — full implementation in Task 3."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PersonaResponse:
    """A single persona's response to a council query."""
    persona_name: str
    content: str
    confidence: float
    dissents: bool
    key_points: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
