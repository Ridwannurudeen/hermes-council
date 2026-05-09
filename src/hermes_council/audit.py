"""Optional local audit logging for council verdicts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def audit_enabled() -> bool:
    return os.getenv("COUNCIL_AUDIT_LOG", "0").lower() in {"1", "true", "yes", "on"}


def get_audit_dir() -> Path:
    configured = os.getenv("COUNCIL_AUDIT_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hermes-council" / "audit"


def write_audit_record(tool: str, request: dict[str, Any], response: dict[str, Any]) -> str | None:
    """Persist an audit record and return the path when logging is enabled."""
    if not audit_enabled():
        return None

    audit_dir = get_audit_dir()
    audit_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    payload = {
        "timestamp": now.isoformat(),
        "tool": tool,
        "request_hash": hashlib.sha256(
            json.dumps(request, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "request": request,
        "response": response,
    }
    safe_tool = _TOOL_FILENAME_RE.sub("_", tool) or "unknown"
    filename = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{safe_tool}.json"
    path = audit_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
