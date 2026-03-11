"""CouncilEvaluator — standalone RL evaluator using adversarial council.

Drop-in evaluator for any RL environment. Uses the council deliberation
engine directly (no MCP overhead).

Usage:
    evaluator = CouncilEvaluator(model="nousresearch/hermes-3-llama-3.1-70b")
    verdict = await evaluator.evaluate(content, question, criteria)
    reward = evaluator.normalized_reward(verdict)
"""

import logging
from typing import Dict, List, Optional

from hermes_council.deliberation import _extract_dpo_pairs, _run_council
from hermes_council.personas import (
    CouncilVerdict,
    PersonaResponse,
    load_custom_personas,
)

logger = logging.getLogger(__name__)


class CouncilEvaluator:
    """Standalone evaluator using the adversarial council for scoring.

    Args:
        model: LLM model name. Defaults to COUNCIL_MODEL env var or hermes-3-70b.
        personas: List of persona names to use. Defaults to all 5.
    """

    def __init__(self, model: str = None, personas: List[str] = None):
        from hermes_council.client import get_model

        self.model = model or get_model()

        all_personas = load_custom_personas()
        if personas:
            self._personas = {
                name.lower(): all_personas[name.lower()]
                for name in personas
                if name.lower() in all_personas
            }
        else:
            self._personas = dict(all_personas)

    async def evaluate(
        self,
        content: str,
        question: str = None,
        criteria: List[str] = None,
    ) -> CouncilVerdict:
        """Run council evaluation on content.

        Args:
            content: The content to evaluate.
            question: The original question/task.
            criteria: Evaluation criteria.

        Returns:
            CouncilVerdict with confidence score and persona responses.
        """
        if criteria is None:
            criteria = ["accuracy", "depth", "evidence", "falsifiability"]

        eval_question = f"Evaluate this content against: {', '.join(criteria)}.\n\n"
        if question:
            eval_question += f"Original task: {question}\n\n"
        eval_question += f"Content:\n{content[:4000]}"

        # Copy persona names to avoid mutation
        persona_names = list(self._personas.keys())

        verdict, meta = await _run_council(
            question=eval_question,
            context="This is an evaluation task. Each persona should critique the content from their intellectual tradition.",
            persona_names=persona_names,
            evidence_search=False,
            model=self.model,
        )

        if verdict is None:
            # Return empty verdict on failure
            return CouncilVerdict(
                question=eval_question,
                responses={},
                arbiter_synthesis="",
                confidence_score=0,
                conflict_detected=False,
            )

        return verdict

    async def gate(self, action: str, context: str = None) -> dict:
        """Quick safety check using Skeptic + Oracle + Arbiter.

        Args:
            action: Description of the action to review.
            context: Why this action is being taken.

        Returns:
            Dict with allowed (bool), confidence (int), reasoning (str).
        """
        from hermes_council.deliberation import _run_gate

        verdict, meta = await _run_gate(
            action=action,
            risk_level="medium",
            context=context or "",
        )

        if verdict is None:
            return {"allowed": False, "confidence": 0, "reasoning": "Evaluation failed"}

        return {
            "allowed": verdict.confidence_score >= 50,
            "confidence": verdict.confidence_score,
            "reasoning": verdict.arbiter_synthesis[:1000],
        }

    def extract_dpo_pairs(self, verdict: CouncilVerdict) -> List[dict]:
        """Extract DPO preference pairs from a verdict."""
        return _extract_dpo_pairs(verdict.question, verdict.responses)

    def normalized_reward(self, verdict: CouncilVerdict) -> float:
        """Convert a verdict to a normalized 0.0-1.0 reward signal."""
        return max(0.0, min(1.0, verdict.confidence_score / 100.0))
