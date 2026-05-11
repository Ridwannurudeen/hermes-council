<div align="center">
  <h1>hermes-council</h1>

  <p><strong>Adversarial preflight and decision review for Hermes Agent.</strong></p>

  <p>
    Hermes Council is an MCP server that lets
    <a href="https://github.com/NousResearch/hermes-agent">Hermes Agent</a>
    stress-test plans, diffs, claims, decisions, and risky actions before it acts.
    It returns structured verdicts, verified evidence snippets, required checks,
    and DPO preference pairs for evaluator and RL workflows.
  </p>

  <p>
    <a href="https://github.com/Ridwannurudeen/hermes-council/actions/workflows/ci.yml">
      <img alt="CI" src="https://github.com/Ridwannurudeen/hermes-council/actions/workflows/ci.yml/badge.svg">
    </a>
    <a href="https://github.com/Ridwannurudeen/hermes-council/actions/workflows/ci.yml">
      <img alt="Coverage" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Ridwannurudeen/316e2a72a4089fdb4fddc9980ed1c28b/raw/coverage.json">
    </a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
    <img alt="MCP" src="https://img.shields.io/badge/MCP-stdio-10B981">
    <a href="LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-111827">
    </a>
  </p>

  <p>
    <a href="#why-it-exists">Why</a> |
    <a href="#what-it-does">What it does</a> |
    <a href="#quickstart">Quickstart</a> |
    <a href="#tools">Tools</a> |
    <a href="#architecture">Architecture</a> |
    <a href="#development">Development</a>
  </p>
</div>

## Why It Exists

Autonomous agents are useful because they act. That is also where they fail.
The failure mode is not usually a syntax error; it is an overconfident plan, a
weak claim, an unsafe command, a diff that looks plausible but breaks a boundary,
or a deployment that should have been stopped by one more adversarial review.

Hermes Council gives Hermes Agent a dedicated judgment layer. Before the agent
ships a plan, changes code, accepts a claim, or takes a risky action, it can call
a council that forces multiple intellectual traditions to argue, dissent, and
produce a structured final verdict.

The result is not another long chain-of-thought prompt. It is an MCP toolset with
explicit verdict fields: `allow`, `allow_with_conditions`, `deny`, `top_risks`,
`required_checks`, `missing_evidence`, `verified_sources`, and `next_actions`.

## What It Does

<table>
  <tr>
    <td><strong>Preflight risky actions</strong><br>Review deploys, migrations, file operations, public messages, and other high-stakes actions before execution. Returns a verdict, blocking risks, required checks, and safer alternatives.</td>
    <td><strong>Review plans and diffs</strong><br>Stress-test implementation plans and code diffs for bugs, missing tests, integration failures, security regressions, and weak assumptions.</td>
  </tr>
  <tr>
    <td><strong>Fact-check claims with evidence</strong><br>Fetch supplied URLs, optionally search the web, pass source snippets to the council, and separate `verified_sources` from model-cited `sources`.</td>
    <td><strong>Compare decisions</strong><br>Evaluate multiple options against explicit criteria and return one recommended path with risks, evidence gaps, and next actions.</td>
  </tr>
  <tr>
    <td><strong>Run adversarial deliberation</strong><br>Use Advocate, Skeptic, Oracle, Contrarian, and Arbiter personas to expose disagreement and synthesize a calibrated verdict.</td>
    <td><strong>Produce RL signals</strong><br>Extract DPO preference pairs and normalized rewards from council verdicts for evaluator and training workflows.</td>
  </tr>
</table>

Supporting features: custom personas, fast/standard/deep modes, optional audit
logs, packaged Hermes skills, OpenAI-compatible provider support, and stdio MCP
transport.

## Quickstart

### Install

```bash
pip install "hermes-council @ git+https://github.com/Ridwannurudeen/hermes-council.git"
```

For RL/evaluator usage:

```bash
pip install "hermes-council[rl] @ git+https://github.com/Ridwannurudeen/hermes-council.git"
```

### Configure Hermes Agent

Add the MCP server to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  council:
    command: python
    args: ["-m", "hermes_council.server"]
```

`hermes-council-server` is also installed as a console script. The `python -m`
form is recommended because it works even when the Python user scripts directory
is not on `PATH`.

You can also wire the server up interactively with Hermes Agent's own CLI:

```bash
hermes mcp add
```

`hermes mcp add/list/remove/test` writes to the same `mcp_servers` key in
`~/.hermes/config.yaml`, so either path produces an equivalent config.

Set one provider key:

```bash
export OPENROUTER_API_KEY=your-key-here
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "your-key-here"
```

Then restart Hermes Agent or run `/reload-mcp`.

### Install Hermes Skills

```bash
hermes-council install-skills
```

This copies skill definitions to `~/.hermes/skills/council/`.

### Smoke Test Without Hermes Agent

```bash
python -m hermes_council.server
```

The server speaks MCP over stdio, so it waits for JSON-RPC messages. Use this
mainly to verify that the module entrypoint imports cleanly.

### Use The Evaluator Directly

```python
import asyncio
from hermes_council.rl.evaluator import CouncilEvaluator


async def main():
    evaluator = CouncilEvaluator(model="nousresearch/hermes-3-llama-3.1-70b")
    verdict = await evaluator.evaluate(
        content="Ship the migration after tests pass.",
        question="Is this deployment plan safe enough?",
        criteria=["safety", "rollback", "evidence"],
    )
    print(verdict.confidence_score)
    print(evaluator.normalized_reward(verdict))


asyncio.run(main())
```

## Tools

Hermes Agent exposes MCP tools with a server prefix. With the config above, the
runtime names are `mcp_council_council_query`,
`mcp_council_council_gate`, and so on.

| Tool | Purpose | Best Use |
|------|---------|----------|
| `council_query` | General adversarial deliberation | Complex questions with no obvious answer |
| `council_evaluate` | Content quality critique | Research, summaries, specs, and generated answers |
| `council_gate` | Safety decision | `allow`, `allow_with_conditions`, or `deny` before action |
| `council_preflight` | Gate with explicit checks | Deployment, migration, irreversible command, public send |
| `council_review_plan` | Plan review | Implementation plan before coding |
| `council_review_diff` | Diff review | Code changes before commit or PR |
| `council_review_claim` | Claim review | Fact-checking with optional evidence retrieval |
| `council_decision` | Option comparison | Pick one path from two or more options |

### Gate Output

```json
{
  "success": true,
  "verdict": "allow_with_conditions",
  "allowed": true,
  "can_proceed_now": false,
  "required_checks": ["verify rollback", "run dry-run"],
  "blocking_risks": ["rollback untested"],
  "safe_alternative": "stage the action first",
  "action_summary": {
    "recommendation": "Proceed after verifying rollback and dry-run output.",
    "top_risks": ["rollback untested"],
    "missing_evidence": ["dry-run output"],
    "next_actions": ["run dry-run", "verify rollback"]
  }
}
```

## Council Modes

| Mode | Calls | Use Case |
|------|-------|----------|
| `fast` | Skeptic + Arbiter | Cheap pre-checks |
| `standard` | Advocate + Skeptic + Oracle + Contrarian + Arbiter | Normal review |
| `deep` | Standard + second Arbiter pass | High-stakes or contentious decisions |

## Architecture

```text
Hermes Agent
    |
    | MCP stdio tool call
    v
hermes-council server
    |
    +--> optional evidence retrieval
    |       - fetch supplied URLs
    |       - optional DuckDuckGo HTML search
    |       - block localhost/private IP targets
    |
    +--> parallel deliberators
    |       - Advocate: steel-man the proposal
    |       - Skeptic: find falsifiers and failure modes
    |       - Oracle: base rates and empirical grounding
    |       - Contrarian: challenge the framing
    |
    +--> Arbiter
            - synthesize disagreement
            - emit structured JSON verdict
            - produce risks, checks, actions, DPO pairs
```

The server uses an OpenAI-compatible async client. It tries JSON mode first and
falls back to text parsing when a provider rejects `response_format`.

## Evidence Model

`evidence_search=true` runs retrieval before persona calls.

| Field | Meaning |
|-------|---------|
| `verified_sources` | URLs actually fetched and summarized by Hermes Council |
| `sources` | URLs cited by model outputs |
| `evidence_errors` | Non-fatal retrieval errors from the evidence layer |

Security boundaries:

- Only `http` and `https` URLs are fetched.
- Localhost, private IPs, link-local IPs, multicast, reserved, and unspecified
  IPs are blocked before fetch.
- Set `COUNCIL_EVIDENCE_SEARCH=0` to disable DuckDuckGo search while still
  allowing supplied public URLs to be fetched.

## Configuration

API key priority is:

```text
COUNCIL_API_KEY > OPENROUTER_API_KEY > NOUS_API_KEY > OPENAI_API_KEY
```

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNCIL_API_KEY` | Council-specific API key | unset |
| `OPENROUTER_API_KEY` | OpenRouter API key | unset |
| `NOUS_API_KEY` | Nous API key | unset |
| `OPENAI_API_KEY` | OpenAI API key | unset |
| `COUNCIL_BASE_URL` | Base URL when `COUNCIL_API_KEY` is used | `https://openrouter.ai/api/v1` |
| `OPENAI_BASE_URL` | Base URL when `OPENAI_API_KEY` is used | `https://api.openai.com/v1` |
| `COUNCIL_MODEL` | Model for persona calls | `nousresearch/hermes-3-llama-3.1-70b` |
| `COUNCIL_TIMEOUT` | LLM request timeout in seconds | `60` |
| `COUNCIL_CONFIG` | Custom persona config path | `~/.hermes-council/config.yaml` |
| `COUNCIL_EVIDENCE_SEARCH` | Enable web search evidence retrieval | `1` |
| `COUNCIL_EVIDENCE_TIMEOUT` | Evidence fetch timeout in seconds | `8` |
| `COUNCIL_AUDIT_LOG` | Write local JSON audit records | `0` |
| `COUNCIL_AUDIT_DIR` | Audit record directory | `~/.hermes-council/audit` |

## Personas

| Persona | Tradition | Role |
|---------|-----------|------|
| Advocate | Steel-manning | Builds the strongest case for the proposal |
| Skeptic | Popperian falsificationism | Finds the observation that would kill the claim |
| Oracle | Empirical base-rate reasoning | Grounds the debate in history and data |
| Contrarian | Kuhnian paradigm critique | Rejects the framing and proposes alternatives |
| Arbiter | Bayesian synthesis | Updates on all arguments and emits the final verdict |

### Custom Personas

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

Custom personas merge with the defaults. Use the same name to override a default.

## Project Layout

```text
hermes-council/
  src/hermes_council/
    server.py          # FastMCP stdio server and public tool handlers
    deliberation.py    # persona orchestration, modes, JSON negotiation, DPO pairs
    evidence.py        # URL/search evidence retrieval and SSRF guards
    audit.py           # optional local JSON verdict logs
    client.py          # provider config and AsyncOpenAI singleton
    personas.py        # default and custom persona definitions
    schemas.py         # Pydantic models for structured model output
    parsing.py         # fallback text parsers for non-JSON providers
    cli.py             # skill installer
    rl/evaluator.py    # direct evaluator API for reward and DPO workflows
  skills/council/      # packaged Hermes skill definitions
  examples/            # Atropos/Ouroboros evaluator example
  tests/               # unit, integration, packaging, and MCP runtime tests
  docs/plans/          # original design and implementation notes
```

## Development

```bash
git clone https://github.com/Ridwannurudeen/hermes-council.git
cd hermes-council
pip install -e ".[dev]"
python -m pytest -q
```

Useful checks:

```bash
python -m pytest -q
python -m ruff check src tests
python -m pytest --cov=hermes_council --cov-report=term-missing -q
python -m pip wheel --no-deps . -w dist
```

## Verification

The test suite covers:

- MCP stdio server startup via `python -m hermes_council.server`
- tool discovery for all council tools
- Hermes-compatible no-key failure behavior
- gate verdict semantics
- evidence retrieval and private-network URL blocking
- packaged skill installation from a wheel
- audit log writing
- custom persona loading
- JSON-mode and fallback parsing paths

## Honest Limitations

- A real provider key is required for actual model-backed verdicts.
- DuckDuckGo HTML search can change; supplied URLs are more reliable than search
  results.
- `verified_sources` proves retrieval, not truth. The Arbiter still weighs the
  evidence.
- The server is stdio MCP only. There is no hosted HTTP service in this repo.
- The council adds latency and token cost. Use `fast` mode for routine preflight.
- Model compliance with JSON fields depends on provider/model behavior, though
  fallback parsing is implemented.

## Roadmap

- [ ] Add a live-provider smoke workflow that runs only when a CI secret is present.
- [ ] Add source ranking and citation-quality scoring for evidence snippets.
- [ ] Add a compact verdict-only mode for very low-latency gates.
- [ ] Add optional HTTP/streamable MCP transport.
- [ ] Add first-class examples for Hermes Agent plan review and diff review sessions.
- [ ] Add benchmark fixtures comparing council review against single-model critique.

## Origin

This project was built in response to feedback on
[hermes-agent PR #848](https://github.com/NousResearch/hermes-agent/pull/848),
where the adversarial council concept was proposed as a core subsystem. The
recommendation was to rebuild it as an external MCP server to avoid core tool
injection, provider bypass, hidden LLM costs, and brittle parsing.

A follow-up integration PR proposing a commented config stub inside hermes-agent
([NousResearch/hermes-agent#1972](https://github.com/NousResearch/hermes-agent/pull/1972))
was **closed on 2026-05-11** — the maintainers do not bundle third-party MCP
server references in `cli-config.yaml.example` or `optional-skills/` to avoid
endorsing one community server over others. Install standalone via the
[Quickstart](#quickstart) above; the server is designed to run that way.

## Contributing

- Keep MCP tool outputs structured and backward-compatible where possible.
- Add tests for every new tool field or runtime boundary.
- Use `python -m hermes_council.server` in docs and integration examples.
- Keep provider-specific behavior behind OpenAI-compatible client settings.
- Run tests, lint, coverage, and wheel packaging checks before opening a PR.

## License

MIT. See [LICENSE](LICENSE).
