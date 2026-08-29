"""Append-only local JSON audit trail for the lab."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import ValidationError

from models import AuditEntry

DEFAULT_AUDIT_LOG_PATH = Path(__file__).with_name("audit_log.json")
_AUDIT_LOCK = threading.Lock()


class AuditLogError(RuntimeError):
    """Raised when the audit history cannot be read or safely updated."""


def read_audit_entries(path: Path | None = None) -> list[AuditEntry]:
    """Read and validate the complete audit history without modifying it."""

    target = path or DEFAULT_AUDIT_LOG_PATH
    if not target.exists():
        return []

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise AuditLogError("Audit log root must be a JSON array.")
        return [AuditEntry.model_validate(item) for item in raw]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise AuditLogError("Audit log is invalid; existing history was not changed.") from exc


def append_audit_entry(entry: AuditEntry, path: Path | None = None) -> None:
    """Append one entry using an atomic replace and preserve existing history."""

    target = path or DEFAULT_AUDIT_LOG_PATH
    temporary = target.with_suffix(".tmp")

    with _AUDIT_LOCK:
        entries = read_audit_entries(target)
        entries.append(entry)
        payload = [item.model_dump(mode="json") for item in entries]
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            raise AuditLogError("Could not append to the audit log.") from exc
