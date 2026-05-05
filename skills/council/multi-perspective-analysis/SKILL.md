---
name: multi-perspective-analysis
description: Run adversarial multi-perspective analysis on complex questions using the council tool. Five personas debate from distinct intellectual traditions.
version: 1.0.0
author: Hermes Ouroboros
license: MIT
metadata:
  hermes:
    tags: [Council, Analysis, Reasoning, Critical-Thinking, Evaluation]
    related_skills: [bayesian-synthesis, adversarial-critique]
---

# Multi-Perspective Analysis

Use the `council_query` tool to submit complex questions for adversarial deliberation by five personas, each representing a distinct intellectual tradition. Use `council_decision` when the user needs a concrete recommendation between options.

## When to Use

- **Complex questions** with no clear right answer
- **High-stakes decisions** where multiple viewpoints matter
- **Research evaluation** to stress-test findings
- **Strategic planning** to surface blind spots
- **Contentious topics** where bias is likely

## The Five Personas

| Persona | Tradition | Role |
|---------|-----------|------|
| **Advocate** | Steel-manning | Builds the strongest case FOR the position |
| **Skeptic** | Popperian falsificationism | Finds the observation that kills the claim |
| **Oracle** | Empirical base-rate reasoning | Grounds debate in historical data and statistics |
| **Contrarian** | Kuhnian paradigm critique | Rejects the framing, proposes alternative paradigms |
| **Arbiter** | Bayesian synthesis | Synthesizes all views with explicit prior/posterior updates |

## How to Use

### Basic Query
```
council_query(question="Should we migrate our monolith to microservices?")
```

### With Context
```
council_query(
    question="Is proof-of-stake more secure than proof-of-work?",
    context="We're evaluating consensus mechanisms for a new L1 blockchain targeting institutional users."
)
```

### Subset of Personas
```
council_query(
    question="Will quantum computing break RSA by 2030?",
    personas=["skeptic", "oracle", "arbiter"]
)
```

### Cost Modes
```
council_query(question="Should we rewrite the storage layer?", mode="fast")
council_query(question="Should we rewrite the storage layer?", mode="standard")
council_query(question="Should we rewrite the storage layer?", mode="deep")
```

- `fast`: Skeptic + Arbiter for cheap pre-checks
- `standard`: Full council for normal decisions
- `deep`: Full council plus a second Arbiter pass for high-stakes decisions

### Decision Review
```
council_decision(
    options=["Ship now", "Delay for audit", "Cut scope and ship"],
    decision_context="Hackathon demo is tomorrow; auth changes are unreviewed",
    criteria=["demo impact", "safety", "reversibility"]
)
```

## Interpreting Results

The council returns a structured verdict:

- **confidence_score** (0-100): Arbiter's posterior confidence after weighing all arguments
- **conflict_detected**: True if personas strongly disagree (confidence spread > 30%)
- **persona_responses**: Each persona's analysis with their individual confidence
- **arbiter_synthesis**: The Arbiter's Bayesian synthesis with prior/posterior reasoning
- **action_summary**: Recommendation, top risks, missing evidence, and next actions
- **dpo_pairs**: Preference pairs for training (aligned vs overruled responses)
- **sources**: Model-cited URLs
- **verified_sources**: URLs actually retrieved by the council evidence layer

### Confidence Guidelines
- **80-100**: Strong consensus, high-quality evidence
- **60-79**: Moderate confidence, some unresolved tensions
- **40-59**: Mixed signals, significant uncertainty
- **0-39**: Low confidence, major disagreements or insufficient evidence

## Workflow: Research Then Evaluate

For best results, combine web research with council evaluation:

1. Use `web_search` to gather initial data
2. Use `web_extract` to read key sources
3. Synthesize findings into a draft analysis
4. Use `council_evaluate` to stress-test your analysis
5. Revise based on council feedback

## Custom Personas

Add custom personas in `~/.hermes-council/config.yaml`:

```yaml
personas:
  security_analyst:
    tradition: "Adversarial security thinking"
    system_prompt: "You are a security analyst. Find every attack vector..."
    tags: ["security", "adversarial"]
```

Custom personas are merged with defaults. Use the same name to override a default.
