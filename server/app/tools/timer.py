"""
Timer tool.

why fire-and-return a confirmation string instead of blocking until the
timer fires: tool calls happen inside a single conversational turn and
must return quickly - the LLM is waiting on this result to keep talking.
The actual "ding" happens on a background asyncio task registered with
the running event loop; see `_notify_after` below.
"""

from __future__ import annotations

import asyncio

SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": "Set a countdown timer for N seconds and notify the user when it elapses.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Duration of the timer in seconds."},
                "label": {"type": "string", "description": "Optional label for what the timer is for."},
            },
            "required": ["seconds"],
        },
    },
}

# populated by pipeline.py at startup with a callback that can push a
# message to the connected websocket client when a timer fires.
_notify_callback = None


def register_notifier(callback) -> None:
    global _notify_callback
    _notify_callback = callback


async def _fire_after(seconds: int, label: str) -> None:
    await asyncio.sleep(seconds)
    if _notify_callback:
        await _notify_callback(label)


def set_timer(seconds: int, label: str = "timer") -> str:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_fire_after(seconds, label))
    except RuntimeError:
        # no running loop (e.g. called from a sync test) - skip scheduling
        pass
    return f"Timer set for {seconds} seconds ({label})."
