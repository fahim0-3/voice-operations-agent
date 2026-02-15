"""Local demonstration tools for security-operations workflows.

They return deterministic records and do not contact real guards, emergency
services, or controllers. Production integrations need authenticated APIs.
"""

from __future__ import annotations

import uuid

from app.audit import write_audit_event


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}}


LOG_WELFARE_CHECK_SCHEMA = _schema("log_welfare_check", "Log a completed welfare check for a guard.", {"guard_id": {"type": "string"}, "status": {"type": "string", "enum": ["ok", "no_response", "needs_follow_up"]}, "notes": {"type": "string"}}, ["guard_id", "status"])
GET_GUARD_STATUS_SCHEMA = _schema("get_guard_status", "Get the demo status for a guard by ID.", {"guard_id": {"type": "string"}}, ["guard_id"])
REPORT_SHIFT_DELAY_SCHEMA = _schema("report_shift_delay", "Report a guard's delayed arrival.", {"guard_id": {"type": "string"}, "minutes_late": {"type": "integer", "minimum": 1}, "reason": {"type": "string"}}, ["guard_id", "minutes_late"])
CREATE_INCIDENT_SCHEMA = _schema("create_incident", "Create an operational incident record.", {"summary": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high"]}, "site": {"type": "string"}}, ["summary", "severity"])
ESCALATE_TO_CONTROLLER_SCHEMA = _schema("escalate_to_controller", "Create a human-controller escalation. This demo does not place calls.", {"reason": {"type": "string"}, "severity": {"type": "string", "enum": ["uncertain", "high", "emergency"]}}, ["reason", "severity"])


def _record(kind: str, **payload: object) -> str:
    record_id = f"{kind[:3].upper()}-{uuid.uuid4().hex[:8]}"
    write_audit_event(kind, record_id=record_id, **payload)
    return f"{kind.replace('_', ' ').title()} recorded: {record_id}."


def log_welfare_check(guard_id: str, status: str, notes: str = "") -> str:
    return _record("welfare_check", guard_id=guard_id, status=status, notes=notes)


def get_guard_status(guard_id: str) -> str:
    write_audit_event("guard_status_lookup", guard_id=guard_id)
    return f"Demo status for {guard_id}: status is not connected to a live guard-management system."


def report_shift_delay(guard_id: str, minutes_late: int, reason: str = "") -> str:
    return _record("shift_delay", guard_id=guard_id, minutes_late=minutes_late, reason=reason)


def create_incident(summary: str, severity: str, site: str = "") -> str:
    return _record("incident", summary=summary, severity=severity, site=site)


def escalate_to_controller(reason: str, severity: str) -> str:
    reference = _record("controller_escalation", reason=reason, severity=severity)
    return f"{reference} Human follow-up is required; no real notification was sent."
