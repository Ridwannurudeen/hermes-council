"""Tests for optional audit logging."""

import json
from pathlib import Path

from hermes_council.audit import audit_enabled, write_audit_record


def test_audit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("COUNCIL_AUDIT_LOG", raising=False)
    assert audit_enabled() is False
    assert write_audit_record("tool", {}, {}) is None


def test_write_audit_record(monkeypatch, tmp_path):
    monkeypatch.setenv("COUNCIL_AUDIT_LOG", "1")
    monkeypatch.setenv("COUNCIL_AUDIT_DIR", str(tmp_path))

    path = write_audit_record(
        "council_gate",
        {"action": "deploy"},
        {"verdict": "allow_with_conditions"},
    )

    assert path is not None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["tool"] == "council_gate"
    assert payload["request_hash"]
    assert payload["response"]["verdict"] == "allow_with_conditions"
