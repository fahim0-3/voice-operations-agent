"""Deterministic intent and escalation policy for guard-operations voice requests."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


OperationalIntent = Literal[
    "log_welfare_check", "get_guard_status", "report_shift_delay", "create_incident", "escalate_to_controller", "unknown"
]


@dataclass(frozen=True)
class PolicyDecision:
    intent: OperationalIntent
    confidence: Literal["high", "low"]
    requires_human_escalation: bool
    reason: str


_EMERGENCY_WORDS = ("emergency", "weapon", "fire", "ambulance", "medical", "attack", "danger", "injured")
_OPERATIONS_WORDS = ("guard", "welfare", "shift", "incident", "controller", "patrol", "site", "safety")
_INTENT_RULES: tuple[tuple[OperationalIntent, tuple[str, ...]], ...] = (
    ("log_welfare_check", ("welfare check", "check welfare", "welfare")),
    (
        "get_guard_status",
        ("guard status", "status of guard", "where is guard", "guard check", "on duty"),
    ),
    ("report_shift_delay", ("shift delay", "running late", "late for shift", "will be late", "delayed")),
    ("create_incident", ("incident", "suspicious", "trespass", "damage", "theft")),
    ("escalate_to_controller", ("escalate", "contact controller", "call controller")),
)


def classify_operational_intent(text: str) -> PolicyDecision:
    """Classify with transparent rules so emergency routing never depends on an LLM."""
    normalized = text.casefold()
    if any(word in normalized for word in _EMERGENCY_WORDS):
        return PolicyDecision("escalate_to_controller", "high", True, "emergency keyword detected")
    # Natural speech often places a duration between "will be" and "late"
    # (for example: "guard G-07 will be 15 minutes late"). Recognize this
    # common shift-delay form without asking the LLM to decide policy.
    if "guard" in normalized and "late" in normalized:
        return PolicyDecision("report_shift_delay", "high", False, "matched guard-delay rule")
    for intent, phrases in _INTENT_RULES:
        if any(phrase in normalized for phrase in phrases):
            return PolicyDecision(intent, "high", intent == "escalate_to_controller", "matched operational rule")
    if any(word in normalized for word in _OPERATIONS_WORDS):
        return PolicyDecision("unknown", "low", True, "operational request could not be classified safely")
    return PolicyDecision("unknown", "low", False, "not an operational request")


def build_confirmed_operation(text: str, intent: OperationalIntent) -> tuple[str, dict[str, object], str]:
    """Build safe demo-tool arguments for a high-confidence policy decision.

    This deliberately handles only the small deterministic operations grammar.
    Other language remains available to the configured LLM provider.
    """
    guard_match = re.search(r"\bG[-\s]?(\d{1,3})\b", text, flags=re.IGNORECASE)
    guard_id = f"G{guard_match.group(1)}" if guard_match else "unidentified-guard"
    normalized = text.casefold()

    if intent == "log_welfare_check":
        status = "no_response" if "no response" in normalized else "needs_follow_up" if "follow up" in normalized else "ok"
        return intent, {"guard_id": guard_id, "status": status, "notes": text}, "Welfare check recorded."
    if intent == "get_guard_status":
        return intent, {"guard_id": guard_id}, "Guard status checked; demo data only."
    if intent == "report_shift_delay":
        minutes_match = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", text, flags=re.IGNORECASE)
        minutes_late = int(minutes_match.group(1)) if minutes_match else 1
        return intent, {"guard_id": guard_id, "minutes_late": minutes_late, "reason": text}, "Shift delay recorded."
    if intent == "create_incident":
        severity = "high" if any(word in normalized for word in ("high", "urgent", "serious")) else "medium"
        return intent, {"summary": text, "severity": severity, "site": ""}, "Incident recorded for human review."
    raise ValueError(f"No deterministic operation builder for intent {intent!r}")