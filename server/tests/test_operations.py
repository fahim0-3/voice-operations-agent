from __future__ import annotations

from app.operations import classify_operational_intent
from app.tools import TOOL_SCHEMAS, execute_tool


def test_emergency_is_deterministically_escalated():
    decision = classify_operational_intent("There is a fire at the site")
    assert decision.intent == "escalate_to_controller"
    assert decision.requires_human_escalation is True


def test_unclear_operational_request_is_escalated():
    decision = classify_operational_intent("The guard situation is strange")
    assert decision.intent == "unknown"
    assert decision.confidence == "low"
    assert decision.requires_human_escalation is True


def test_known_welfare_intent_is_not_automatically_escalated():
    decision = classify_operational_intent("Please log a welfare check for G-12")
    assert decision.intent == "log_welfare_check"
    assert decision.requires_human_escalation is False


def test_guard_status_spoken_in_natural_word_order_is_recognized():
    decision = classify_operational_intent("What is the status of guard G-21?")
    assert decision.intent == "get_guard_status"


def test_guard_delay_with_duration_is_recognized():
    decision = classify_operational_intent("Guard G-07 will be 15 minutes late because of traffic")
    assert decision.intent == "report_shift_delay"


def test_operations_tools_are_registered_and_deterministic():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert {"log_welfare_check", "get_guard_status", "report_shift_delay", "create_incident", "escalate_to_controller"} <= names
    result = execute_tool("report_shift_delay", {"guard_id": "G-12", "minutes_late": 10})
    assert "recorded" in result.lower()
