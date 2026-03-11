# hermes-council MCP Server — Design Document

**Date**: 2026-03-11
**Origin**: NousResearch/hermes-agent PR #848 review feedback from teknium1
**PR link**: https://github.com/NousResearch/hermes-agent/pull/848

## Background

The adversarial council subsystem was submitted as a core integration into hermes-agent. The maintainer (teknium1) rejected core inclusion for four reasons:

1. **Core tool bloat** — `council_query` was added to `_HERMES_CORE_TOOLS`, injected into every session
2. **Provider bypass** — The tool created its own `AsyncOpenAI` client, ignoring the agent's configured provider chain
3. **Hidden cost** — 5 LLM calls per invocation, invisible to users
4. **Brittle regex parsing** — Confidence/dissent parsed via regex from free-form text, silent fallback to defaults

The maintainer recommended restructuring as a standalone MCP server package. This document specifies that package.

## Package Structure

```
hermes-council/
├── pyproject.toml
├── config.example.yaml
├── src/hermes_council/
│   ├── __init__.py                   # Version, public API exports
│   ├── server.py                     # FastMCP stdio server entry point
│   ├── personas.py                   # Persona dataclass + 5 defaults + custom loading
│   ├── deliberation.py               # Core orchestration: _run_council(), _run_gate()
│   ├── schemas.py                    # Pydantic models for JSON mode responses
│   ├── client.py                     # Lazy singleton AsyncOpenAI + config resolution
│   ├── parsing.py                    # Regex fallback parsers (for non-JSON-mode providers)
│   ├── cli.py                        # hermes-council CLI (install-skills)
│   └── rl/
│       ├── __init__.py
│       └── evaluator.py              # CouncilEvaluator (standalone, no hermes-agent deps)
├── examples/
│   ├── ouroboros_env.py              # OuroborosEnv template (copy into hermes-agent)
│   └── ouroboros.yaml                # Datagen config for Atropos
├── skills/council/
│   ├── DESCRIPTION.md
│   ├── multi-perspective-analysis/SKILL.md
│   ├── bayesian-synthesis/SKILL.md
│   └── adversarial-critique/SKILL.md
├── tests/
│   ├── test_personas.py
│   ├── test_schemas.py
│   ├── test_deliberation.py
│   ├── test_server.py
│   ├── test_client.py
│   └── test_evaluator.py
├── LICENSE                            # MIT (matches hermes-agent)
└── README.md
```

### Entry Points

```toml
[project.scripts]
hermes-council-server = "hermes_council.server:main"
hermes-council = "hermes_council.cli:main"
```

### User Installation

```bash
pip install hermes-council
```

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  council:
    command: hermes-council-server
```

Tools appear automatically in the next hermes session.

## Dependencies

```toml
[project]
dependencies = [
    "mcp>=1.2.0",
    "openai>=1.6.0",
    "pydantic>=2.0",
    "pyyaml",
]

[project.optional-dependencies]
rl = ["openai>=1.6.0", "pyyaml", "pydantic>=2.0"]
dev = ["pytest", "pytest-asyncio"]
```

The `rl` extra installs just enough to use `CouncilEvaluator` as a library without the MCP server. `atroposlib` and hermes-agent are NOT dependencies — `OuroborosEnv` ships as an example template users copy into their hermes-agent checkout.

## Client & Config (`client.py`)

### API Key Resolution

Priority order:

| Priority | Key env var | Base URL env var | Default base URL |
|---|---|---|---|
| 1 | `COUNCIL_API_KEY` | `COUNCIL_BASE_URL` | `https://openrouter.ai/api/v1` |
| 2 | `OPENROUTER_API_KEY` | — | `https://openrouter.ai/api/v1` |
| 3 | `NOUS_API_KEY` | — | `https://inference-api.nousresearch.com/v1` |
| 4 | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |

### Model

`COUNCIL_MODEL` env var, default: `nousresearch/hermes-3-llama-3.1-70b`

### Timeout

`COUNCIL_TIMEOUT` env var, default: `60` (seconds per LLM call). Passed as `timeout=httpx.Timeout(value)` to `AsyncOpenAI()`.

### Client Lifecycle

**Lazy singleton** — client is created on first LLM call, not at module import. This avoids misconfiguration if env vars aren't set at import time. Once created, the instance is cached and reused for all subsequent calls within the server process.

If no API key resolves, tools return a clear error: `"No API key configured. Set COUNCIL_API_KEY, OPENROUTER_API_KEY, NOUS_API_KEY, or OPENAI_API_KEY."`

### CouncilEvaluator Client

The RL evaluator maintains its own separate lazy client instance (not shared with the MCP server singleton). This allows library use without the MCP server running.

## Personas (`personas.py`)

### Dataclasses

Carried over from original:

```python
@dataclass
class Persona:
    name: str
    tradition: str
    system_prompt: str
    scoring_weights: Dict[str, float]  # Preserved
    tags: List[str]                    # Preserved

@dataclass
class PersonaResponse:
    persona_name: str
    content: str
    confidence: float       # 0.0-1.0
    dissents: bool
    key_points: List[str]
    sources: List[str]

@dataclass
class CouncilVerdict:
    question: str
    responses: Dict[str, PersonaResponse]
    arbiter_synthesis: str
    confidence_score: int   # 0-100
    conflict_detected: bool
    dpo_pairs: List[Dict]
    sources: List[str]
```

### 5 Default Personas

All carry over with their intellectual traditions:

- **Advocate** — Steel-manning (strongest case FOR)
- **Skeptic** — Popperian falsificationism (find the killing observation)
- **Oracle** — Empirical base-rate reasoning (historical data + base rates)
- **Contrarian** — Kuhnian paradigm critique (reject the framing)
- **Arbiter** — Bayesian synthesis (prior → evidence updates → posterior)

### System Prompt Changes

Each persona's system prompt is rewritten to end with a JSON output instruction:

**Deliberator personas** (Advocate, Skeptic, Oracle, Contrarian):

```
You MUST respond in valid JSON with these exact keys:
{
  "reasoning": "your full analysis text",
  "confidence": <float 0.0-1.0>,
  "dissent": <true|false>,
  "key_points": ["point 1", "point 2"],
  "sources": ["url1", "url2"]
}
Do not include any text outside the JSON object.
```

**Arbiter** (additional keys):

```
You MUST respond in valid JSON with these exact keys:
{
  "reasoning": "your full analysis text",
  "confidence": <float 0.0-1.0>,
  "dissent": false,
  "key_points": ["point 1", "point 2"],
  "sources": ["url1", "url2"],
  "prior": "your starting belief before reading arguments",
  "posterior": "your updated belief after evidence",
  "evidence_updates": ["Advocate: +X% because...", "Skeptic: -Y% because..."],
  "risk_level": "low|medium|high|critical",
  "consensus": "clear recommendation in 2-3 sentences"
}
Do not include any text outside the JSON object.
```

The rest of each persona's system prompt (intellectual tradition, guidelines) stays the same.

### Custom Persona Loading

Config path: `COUNCIL_CONFIG` env var, or default `~/.hermes-council/config.yaml`.

Does NOT read `~/.hermes/config.yaml` (that belongs to hermes-agent).

YAML structure:

```yaml
personas:
  researcher:
    tradition: "Systematic literature review"
    system_prompt: "You are the Researcher..."
    scoring_weights:
      evidence: 0.4
      methodology: 0.3
      reproducibility: 0.2
      novelty: 0.1
    tags: ["research", "systematic", "evidence"]
```

Custom personas merge with (and can override) defaults.

## Structured Output (`schemas.py`)

Pydantic models for validating JSON mode responses:

```python
class PersonaOutput(BaseModel):
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    dissent: bool = False
    key_points: list[str] = []
    sources: list[str] = []

class ArbiterOutput(PersonaOutput):
    prior: str = ""
    posterior: str = ""
    evidence_updates: list[str] = []
    risk_level: str = "medium"
    consensus: str = ""

class DPOPair(BaseModel):
    question: str
    chosen: str
    rejected: str
    confidence: float
    source: str = "council_evaluation"
    chosen_persona: str
    rejected_persona: str
```

### JSON Mode Fallback

LLM calls use `response_format={"type": "json_object"}`. If the provider rejects this parameter (400 error):

1. Retry the call WITHOUT `response_format`
2. Set a module-level flag `_json_mode_supported = False`
3. All subsequent calls skip `response_format`
4. Parse free-text responses using regex fallback (`parsing.py`)
5. Log a warning to stderr: `"JSON mode not supported by provider, falling back to text parsing"`

The regex fallback parsers are the original parsers from the PR, moved to `parsing.py`:

- `_parse_confidence()` — extracts float from "CONFIDENCE: 0.85" or "85%"
- `_parse_dissent()` — extracts bool from "DISSENT: true/false"
- `_parse_key_points()` — extracts bullet points
- `_extract_sources()` — extracts URLs

On malformed JSON (model returns JSON mode but invalid structure): wrap raw text in `PersonaOutput(reasoning=raw_text, confidence=0.5, dissent=False)` and log a warning. This is the same silent default as the original — but with a logged warning so it's visible in stderr.

## Deliberation (`deliberation.py`)

### `_run_council(question, context, persona_names, evidence_search, model)`

1. Load personas via `load_custom_personas()`. **Copy the dict** before popping arbiter — never mutate the source.
2. Build user message from question + context. If `evidence_search=True`, append: `"Note: If you have access to web search, use it to find supporting evidence or counter-evidence for your analysis."`
3. Run deliberators in parallel via `asyncio.gather(*tasks, return_exceptions=True)`.
4. **Graceful degradation**: skip failed personas, log errors to stderr. If ALL deliberators fail (0 valid responses), return error immediately — do not run Arbiter.
5. Detect conflict: confidence spread > 0.3 among deliberators.
6. Build Arbiter context from deliberator responses. **Truncate each deliberator's response to 3000 chars** to prevent token overflow with many custom personas.
7. Run Arbiter with synthesized context.
8. Aggregate sources, compute confidence score (`int(arbiter.confidence * 100)`).
9. Extract DPO pairs.
10. Track token usage: accumulate `response.usage.total_tokens` from each LLM call.
11. Return `CouncilVerdict` + metadata dict with `calls_made`, `model`, `total_tokens`.

### `_run_gate(action, risk_level, context)`

Shorthand: calls `_run_council(persona_names=["skeptic", "oracle", "arbiter"], evidence_search=False)`.

### Timeout

Each LLM call respects `COUNCIL_TIMEOUT` (default 60s) via the `AsyncOpenAI` client timeout. The entire deliberation is wrapped in `asyncio.wait_for(total_timeout)` where `total_timeout = COUNCIL_TIMEOUT * 2.5` (allows for 5 sequential worst-case calls, but parallel execution means it's usually much faster).

## MCP Server (`server.py`)

FastMCP stdio server. Entry point: `hermes-council-server`.

### Logging

All logging goes to stderr:

```python
import sys, logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
```

stdout is reserved exclusively for the MCP JSON-RPC protocol.

### Tool 1: `council_query`

**Params:**
- `question` (string, required) — The question to deliberate on
- `context` (string, optional) — Prior research or context
- `personas` (array of strings, optional) — Subset of persona names (default: all 5)
- `evidence_search` (boolean, optional, default: true) — Encourage evidence citation

**Response:**
```json
{
  "success": true,
  "question": "...",
  "confidence_score": 78,
  "conflict_detected": false,
  "arbiter_synthesis": "...",
  "persona_responses": {
    "advocate": {
      "confidence": 0.85,
      "dissents": false,
      "key_points": ["..."],
      "content": "...(truncated to 2000 chars)",
      "sources": ["..."]
    }
  },
  "dpo_pairs": [...],
  "sources": ["..."],
  "available_personas": ["advocate", "skeptic", "oracle", "contrarian", "arbiter"],
  "_meta": {
    "calls_made": 5,
    "model": "nousresearch/hermes-3-llama-3.1-70b",
    "total_tokens": 4231
  }
}
```

### Tool 2: `council_evaluate`

**Params:**
- `content` (string, required) — Content to evaluate
- `question` (string, optional) — Original task the content addresses
- `criteria` (array of strings, optional, default: `["accuracy", "depth", "evidence", "falsifiability"]`)

**Response:** Same structure as `council_query`, with `persona_responses` content truncated to 1500 chars and `criteria` field added.

`_meta.calls_made` = 5.

### Tool 3: `council_gate`

**Params:**
- `action` (string, required) — Description of the action to review
- `risk_level` (string, optional, enum: low/medium/high, default: medium)
- `context` (string, optional) — Why the action is being taken

**Response:**
```json
{
  "success": true,
  "allowed": true,
  "confidence": 72,
  "risk_level": "medium",
  "threshold": 50,
  "reasoning": "...(truncated to 1000 chars)",
  "skeptic_concerns": ["...", "..."],
  "_meta": {
    "calls_made": 3,
    "model": "nousresearch/hermes-3-llama-3.1-70b",
    "total_tokens": 2100
  }
}
```

**Thresholds:** low=30, medium=50, high=70.

### Error Responses

All tools return errors in a consistent format:

```json
{
  "success": false,
  "error": "descriptive message"
}
```

Error cases:
- No API key configured
- Missing required parameter
- All deliberators failed
- Deliberation timed out

## CLI (`cli.py`)

Single command via `argparse`:

```bash
hermes-council install-skills [--force]
```

- Copies `skills/council/` from the installed package to `~/.hermes/skills/council/`
- If target exists and `--force` not passed: prints warning and exits
- If `--force`: overwrites existing skills
- Uses `importlib.resources` to locate bundled skill files
- Prints confirmation with paths created

## RL Components

### `CouncilEvaluator` (`rl/evaluator.py`)

Standalone evaluator that imports from `hermes_council.deliberation` and `hermes_council.personas` directly. No MCP overhead, no hermes-agent dependency.

**Key changes from original:**
- Import paths updated to `hermes_council.*`
- **Dict mutation fix**: `evaluate()` copies `self._personas` before popping arbiter (`dict(self._personas)`), never mutates instance state
- Own lazy `AsyncOpenAI` client (separate from MCP server singleton)
- Uses JSON mode with same fallback logic as server

**API (unchanged):**
- `evaluate(content, question, criteria)` → `CouncilVerdict`
- `gate(action, context)` → `{"allowed": bool, "confidence": int, "reasoning": str}`
- `normalized_reward(verdict)` → float 0.0-1.0
- `extract_dpo_pairs(verdict)` → list of DPO dicts

### `OuroborosEnv` (`examples/ouroboros_env.py`)

Ships as an **example template**, NOT importable library code. Users copy it into their hermes-agent `environments/` directory.

Reason: OuroborosEnv imports `HermesAgentBaseEnv`, `AgentResult`, `ToolContext`, `APIServerConfig` from hermes-agent internals. Making hermes-agent a dependency of hermes-council would create a circular dependency.

Instructions in README:
```bash
cp $(python -c "import hermes_council; print(hermes_council.__path__[0])")/../examples/ouroboros_env.py \
   /path/to/hermes-agent/environments/
```

## Skills

Three skill markdown files carry over unchanged:

- `multi-perspective-analysis/SKILL.md` — Guide for `council_query`
- `bayesian-synthesis/SKILL.md` — Arbiter's methodology
- `adversarial-critique/SKILL.md` — Stress-testing with `council_evaluate` and `council_gate`

Installed via `hermes-council install-skills` into `~/.hermes/skills/council/`.

## Testing

Target: maintain original 53+ test count.

| Test file | Covers | Tests from original |
|---|---|---|
| `test_personas.py` | Default loading, fields, case-insensitive lookup, scoring_weights sum, tags, custom YAML loading (new config path) | TestCouncilPersonas |
| `test_schemas.py` | PersonaOutput/ArbiterOutput/DPOPair validation, JSON parsing, malformed JSON fallback, regex fallback path | NEW |
| `test_deliberation.py` | Mocked LLM for all 3 flows, graceful degradation (1 fails, all fail), conflict detection, DPO extraction, token accumulation, evidence_search flag, response truncation | TestCouncilHandlers + TestDPOExtraction + TestResponseParsing |
| `test_client.py` | Key priority resolution (COUNCIL > OPENROUTER > NOUS > OPENAI > none), lazy init, timeout config | TestAPIConfig |
| `test_server.py` | FastMCP tool registration, 3 tools exposed, schema structure, missing param errors, _meta in responses | TestCouncilRegistration |
| `test_evaluator.py` | CouncilEvaluator init (default, custom, persona subset), normalized_reward (clamped), dict mutation safety (concurrent calls), extract_dpo_pairs | TestCouncilEvaluator |

All LLM calls mocked. `pytest` + `pytest-asyncio`.

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `COUNCIL_API_KEY` | — | API key (highest priority) |
| `COUNCIL_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL (used with COUNCIL_API_KEY) |
| `COUNCIL_MODEL` | `nousresearch/hermes-3-llama-3.1-70b` | Model for all personas |
| `COUNCIL_TIMEOUT` | `60` | Seconds per LLM call |
| `COUNCIL_CONFIG` | `~/.hermes-council/config.yaml` | Custom persona config path |
| `OPENROUTER_API_KEY` | — | Fallback key (priority 2) |
| `NOUS_API_KEY` | — | Fallback key (priority 3) |
| `OPENAI_API_KEY` | — | Fallback key (priority 4) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Used with OPENAI_API_KEY |

## Design Decisions Log

| # | Decision | Rationale |
|---|---|---|
| 1 | Standalone MCP server, not core integration | Maintainer feedback: no core bloat, tools appear only when configured |
| 2 | stdio transport only | Matches existing hermes MCP servers, lowest friction, HTTP can be added later |
| 3 | `response_format` JSON mode with regex fallback | Fixes brittle parsing (teknium #4) while maintaining broad provider compatibility |
| 4 | Lazy singleton client | Avoids misconfiguration at import time, safe for MCP stdio lifecycle |
| 5 | NOUS_API_KEY in fallback chain | NousResearch is the parent project — don't drop their key |
| 6 | `~/.hermes-council/config.yaml` for custom personas | Standalone config, doesn't read hermes-agent's config |
| 7 | OuroborosEnv as example template, not library | Avoids circular dependency with hermes-agent internals |
| 8 | Dict copy before arbiter pop | Fixes race condition in original CouncilEvaluator |
| 9 | `_meta` block in all tool responses | Fixes hidden cost (teknium #3): calls_made, model, total_tokens visible |
| 10 | stderr-only logging | stdout is MCP JSON-RPC protocol for stdio servers |
| 11 | All-fail error path | Returns structured error instead of empty/meaningless verdict |
| 12 | Arbiter context truncation (3000 chars/persona) | Prevents token overflow with many custom personas |
