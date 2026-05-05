"""Runtime tests for the packaged stdio MCP server."""

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_python_module_stdio_server_exposes_council_tools():
    env = os.environ.copy()
    for key in ["COUNCIL_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY"]:
        env.pop(key, None)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hermes_council.server"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = {tool.name for tool in listed.tools}

                assert {
                    "council_query",
                    "council_evaluate",
                    "council_gate",
                    "council_preflight",
                    "council_review_plan",
                    "council_review_diff",
                    "council_review_claim",
                    "council_decision",
                }.issubset(names)

                result = await session.call_tool("council_gate", {"action": "Deploy"})
                payload = json.loads(result.content[0].text)

    assert payload["success"] is False
    assert "No API key" in payload["error"]
    assert payload["calls_made"] == 0
