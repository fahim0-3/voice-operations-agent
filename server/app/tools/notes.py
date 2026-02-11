"""
Notes tool - simple persistent scratchpad memory ("remember that ...").

why a flat JSON file instead of a real database: this is a portfolio demo
of agent tool-use patterns, not a note-taking product - a SQLite/vector
store would be over-engineering here. The interesting part is the
tool-calling contract, not the storage backend (swap this file for a real
DB in five minutes if you needed to).
"""

from __future__ import annotations

import json
from pathlib import Path

_NOTES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "notes.json"

SAVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a short note/reminder for later recall. Use when the user says 'remember...' or 'note that...'.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note content to remember."}},
            "required": ["text"],
        },
    },
}

RECALL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_notes",
        "description": "Recall previously saved notes. Optionally filter by a keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Optional keyword to filter notes by."}
            },
            "required": [],
        },
    },
}


def _load() -> list[dict]:
    if not _NOTES_PATH.exists():
        return []
    return json.loads(_NOTES_PATH.read_text(encoding="utf-8"))


def _save(notes: list[dict]) -> None:
    _NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _NOTES_PATH.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def save_note(text: str) -> str:
    import datetime

    notes = _load()
    notes.append({"text": text, "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _save(notes)
    return f"Noted: '{text}'."


def recall_notes(keyword: str = "") -> str:
    notes = _load()
    if keyword:
        notes = [n for n in notes if keyword.lower() in n["text"].lower()]
    if not notes:
        return "No matching notes found."
    return "; ".join(f"[{n['ts']}] {n['text']}" for n in notes[-5:])
