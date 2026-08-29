from __future__ import annotations

from pathlib import Path

import pytest

from audit import AuditLogError, append_audit_entry, read_audit_entries
from models import AuditEntry


def _entry(decision: str) -> AuditEntry:
    return AuditEntry(
        timestamp="2026-08-29T09:00:00+00:00",
        agent_id="churn-risk-agent",
        action="send_email",
        confidence=0.9,
        reviewer_id="operator_01",
        decision=decision,
    )


def test_append_preserves_existing_history(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"

    append_audit_entry(_entry("approve"), audit_path)
    append_audit_entry(_entry("reject"), audit_path)

    entries = read_audit_entries(audit_path)
    assert [entry.decision for entry in entries] == ["approve", "reject"]


def test_invalid_history_fails_closed_without_overwrite(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    original = "not valid json"
    audit_path.write_text(original, encoding="utf-8")

    with pytest.raises(AuditLogError):
        append_audit_entry(_entry("approve"), audit_path)

    assert audit_path.read_text(encoding="utf-8") == original
