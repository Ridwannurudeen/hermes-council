"""Pydantic models for structured JSON mode responses."""

from pydantic import BaseModel, Field


class PersonaOutput(BaseModel):
    """Structured output from a deliberator persona."""

    model_config = {"extra": "ignore"}

    reasoning: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    dissent: bool = False
    key_points: list[str] = []
    sources: list[str] = []


class ArbiterOutput(PersonaOutput):
    """Structured output from the Arbiter persona (extends PersonaOutput)."""

    prior: str = ""
    posterior: str = ""
    evidence_updates: list[str] = []
    risk_level: str = "medium"
    consensus: str = ""


class DPOPair(BaseModel):
    """A DPO preference pair extracted from council disagreement."""

    question: str
    chosen: str
    rejected: str
    confidence: float
    source: str = "council_evaluation"
    chosen_persona: str
    rejected_persona: str
