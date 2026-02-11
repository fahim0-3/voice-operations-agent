"""Small JSON-lines audit trail for operational decisions and tool activity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings


def write_audit_event(event: str, *, turn_id: str | None = None, **details: Any) -> None:
    """Append a structured event locally; deliberately no external database is required."""
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "turn_id": turn_id,
        **details,
    }
    with path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
