# hermes-council

Adversarial multi-perspective council MCP server for [hermes-agent](https://github.com/NousResearch/hermes-agent).

Five personas from distinct intellectual traditions debate questions through structured adversarial deliberation, producing calibrated verdicts with confidence scores, evidence links, and DPO preference pairs for RL training.

## Install

```bash
pip install git+https://github.com/Ridwannurudeen/hermes-council.git
```

For RL components:
```bash
pip install "hermes-council[rl] @ git+https://github.com/Ridwannurudeen/hermes-council.git"
```

## Quick Start

Add to your hermes-agent config (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  council:
    command: hermes-council-server
```

Set your API key:
```bash
export OPENROUTER_API_KEY=your-key-here
```

## Tools

| Tool | Description | Personas |
|------|-------------|----------|
| `council_query` | Full 5-persona adversarial deliberation on complex questions | All 5 |
| `council_evaluate` | Evaluate content quality through adversarial critique | All 5 |
| `council_gate` | Quick safety review before high-stakes actions | Skeptic + Oracle + Arbiter |

## Configuration

All configuration is via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNCIL_API_KEY` | API key (highest priority) | — |
| `OPENROUTER_API_KEY` | OpenRouter API key | — |
| `NOUS_API_KEY` | Nous API key | — |
| `OPENAI_API_KEY` | OpenAI API key (lowest priority) | — |
| `COUNCIL_BASE_URL` | API base URL | `https://openrouter.ai/api/v1` |
| `COUNCIL_MODEL` | Model to use | `nousresearch/hermes-3-llama-3.1-70b` |
| `COUNCIL_TIMEOUT` | Request timeout (seconds) | `60` |
| `COUNCIL_CONFIG` | Path to custom config YAML | `~/.hermes-council/config.yaml` |

## Custom Personas

Create `~/.hermes-council/config.yaml`:

```yaml
personas:
  security_analyst:
    tradition: "Adversarial security thinking"
    system_prompt: "You are a security analyst. Evaluate every claim for attack vectors, failure modes, and adversarial scenarios."
    scoring_weights:
      threat_assessment: 0.4
      evidence: 0.3
      rigor: 0.3
    tags: ["security", "adversarial"]
```

Custom personas merge with the 5 defaults. Use the same name to override a default.

## The Five Personas

| Persona | Tradition | Role |
|---------|-----------|------|
| **Advocate** | Steel-manning | Builds the strongest case FOR the position |
| **Skeptic** | Popperian falsificationism | Finds the observation that kills the claim |
| **Oracle** | Empirical base-rate reasoning | Grounds debate in historical data and statistics |
| **Contrarian** | Kuhnian paradigm critique | Rejects the framing, proposes alternative paradigms |
| **Arbiter** | Bayesian synthesis | Synthesizes all views with explicit prior/posterior updates |

## Skills

Install council skills for hermes-agent:

```bash
hermes-council install-skills
```

This copies skill definitions to `~/.hermes/skills/council/`:
- `multi-perspective-analysis` — Full council deliberation workflow
- `bayesian-synthesis` — Bayesian reasoning and evidence synthesis
- `adversarial-critique` — Stress-testing and safety gating

## RL Integration

### CouncilEvaluator

Use the council as a reward signal for RL training:

```python
from hermes_council.rl.evaluator import CouncilEvaluator

evaluator = CouncilEvaluator(model="nousresearch/hermes-3-llama-3.1-70b")

# Evaluate agent output
verdict = await evaluator.evaluate(
    content=agent_output,
    question="Research question here",
    criteria=["accuracy", "depth", "evidence", "falsifiability"],
)

# Get normalized reward (0.0-1.0)
reward = evaluator.normalized_reward(verdict)

# Extract DPO preference pairs
dpo_pairs = evaluator.extract_dpo_pairs(verdict)
```

### OuroborosEnv

An example Atropos RL environment is provided in `examples/ouroboros_env.py`. Copy it into your hermes-agent installation:

```bash
cp examples/ouroboros_env.py /path/to/hermes-agent/environments/
cp examples/ouroboros.yaml /path/to/hermes-agent/datagen-config-examples/
```

## Development

```bash
git clone https://github.com/Ridwannurudeen/hermes-council.git
cd hermes-council
pip install -e ".[dev]"
pytest
```

## License

MIT
