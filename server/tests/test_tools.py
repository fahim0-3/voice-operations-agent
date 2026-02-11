"""
Tool registry tests - the registry dispatch, sysinfo, notes, and the
app-launcher allowlist security boundary.
"""

from __future__ import annotations

from app.tools import TOOL_SCHEMAS, execute_tool

from app.tools.media import launch_app
from app.tools.sysinfo import get_system_stats


def test_all_schemas_have_required_openai_shape():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_execute_tool_unknown_name_does_not_raise():
    result = execute_tool("not_a_real_tool", {})
    assert "unknown tool" in result.lower()


def test_execute_tool_bad_arguments_does_not_raise():
    # get_system_stats takes no args - passing one should be caught, not crash the turn
    result = execute_tool("get_system_stats", {"bogus_arg": 1})
    assert isinstance(result, str)


def test_get_system_stats_returns_cpu_and_ram_info():
    result = get_system_stats()
    assert "CPU" in result
    assert "RAM" in result


def test_launch_app_rejects_non_allowlisted_app():
    """
    why this test matters: this IS the security boundary described in
    media.py's docstring. If this regresses, launch_app becomes a
    potential arbitrary-command-execution vector for a prompt-injected
    tool call.
    """
    result = launch_app("powershell -Command Remove-Item C:\\")
    assert "not in the allowed app list" in result


def test_notes_save_and_recall(monkeypatch, tmp_path):
    from app.tools import notes

    fake_path = tmp_path / "notes.json"
    monkeypatch.setattr(notes, "_NOTES_PATH", fake_path)

    notes.save_note("buy milk")
    notes.save_note("call the dentist")

    result = notes.recall_notes()
    assert "buy milk" in result
    assert "call the dentist" in result

    filtered = notes.recall_notes(keyword="dentist")
    assert "dentist" in filtered
    assert "milk" not in filtered
