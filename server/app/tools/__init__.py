"""
Tool registry.

why a registry pattern (dict of name -> function) instead of if/elif chains
in llm.py: adding a new tool should mean "write one function + one schema
entry", not "hunt through llm.py for the right if-branch to edit". This
also means execute_tool() is trivially unit-testable in isolation from any
LLM/network code.

Each tool module exports:
  - a JSON schema dict (Ollama/OpenAI "function calling" format)
  - a plain Python function that takes the parsed arguments and returns a
    string (tool results are always fed back to the LLM as text)

See docs/07-tool-calling.md for the full contract and security notes on
the app-launcher allowlist.
"""

from __future__ import annotations

from typing import Any

from app.tools import media, notes, operations, sysinfo, timer, websearch

TOOL_SCHEMAS: list[dict] = [
    sysinfo.SCHEMA,
    timer.SCHEMA,
    notes.SAVE_SCHEMA,
    notes.RECALL_SCHEMA,
    media.LAUNCH_SCHEMA,
    media.VOLUME_SCHEMA,
    websearch.SCHEMA,
    operations.LOG_WELFARE_CHECK_SCHEMA,
    operations.GET_GUARD_STATUS_SCHEMA,
    operations.REPORT_SHIFT_DELAY_SCHEMA,
    operations.CREATE_INCIDENT_SCHEMA,
    operations.ESCALATE_TO_CONTROLLER_SCHEMA,
]

_HANDLERS = {
    "get_system_stats": sysinfo.get_system_stats,
    "set_timer": timer.set_timer,
    "save_note": notes.save_note,
    "recall_notes": notes.recall_notes,
    "launch_app": media.launch_app,
    "set_volume": media.set_volume,
    "web_search": websearch.web_search,
    "log_welfare_check": operations.log_welfare_check,
    "get_guard_status": operations.get_guard_status,
    "report_shift_delay": operations.report_shift_delay,
    "create_incident": operations.create_incident,
    "escalate_to_controller": operations.escalate_to_controller,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Dispatch a tool call by name. Never raises - tool errors are returned
    as a string so the LLM can react gracefully ("I couldn't find that
    app, sir") instead of the whole turn crashing.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'."
    try:
        return handler(**arguments)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
        return f"Error running tool '{name}': {exc}"
