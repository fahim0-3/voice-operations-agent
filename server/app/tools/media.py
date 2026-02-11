"""
App launcher + volume control.

why an ALLOWLIST for launch_app instead of running whatever string the LLM
gives us: the LLM's tool arguments are effectively untrusted input (a
prompt-injection in a webpage/note the agent reads could try to make it
"launch" an arbitrary command). Restricting to a fixed name->path map
means the worst case is "it opens the wrong allowed app", not arbitrary
code execution. See docs/07-tool-calling.md "security notes".
"""

from __future__ import annotations

import subprocess

LAUNCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "launch_app",
        "description": "Launch a known desktop application by name (e.g. 'notepad', 'browser', 'calculator').",
        "parameters": {
            "type": "object",
            "properties": {"app": {"type": "string", "description": "Name of the app to launch."}},
            "required": ["app"],
        },
    },
}

VOLUME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_volume",
        "description": "Set the system volume to a percentage (0-100), or mute/unmute.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Target volume percentage, 0-100. Omit if using 'mute'/'unmute' action instead.",
                },
                "action": {
                    "type": "string",
                    "enum": ["set", "mute", "unmute"],
                    "description": "What to do. Default 'set' uses 'level'.",
                },
            },
            "required": [],
        },
    },
}

# Explicit allowlist - see module docstring for why this can't be arbitrary.
_APP_ALLOWLIST: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "browser": "cmd /c start msedge",
    "explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "terminal": "wt.exe",
}


def launch_app(app: str) -> str:
    key = app.strip().lower()
    target = _APP_ALLOWLIST.get(key)
    if not target:
        allowed = ", ".join(sorted(_APP_ALLOWLIST))
        return f"'{app}' is not in the allowed app list. Allowed: {allowed}."
    try:
        subprocess.Popen(target, shell=True)  # noqa: S602 - target comes only from our fixed allowlist above
        return f"Launched {app}."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to launch {app}: {exc}"


def set_volume(level: int | None = None, action: str = "set") -> str:
    """
    Windows-only volume control via the Core Audio API (pycaw).

    why pycaw instead of SendKeys/media-key simulation: SendKeys nudges
    volume in fixed ~2% steps and has no way to set an *exact* target
    percentage or query current volume - it's a hack that drifts. pycaw
    talks to the real Windows Core Audio (IAudioEndpointVolume) API, so
    "set volume to 30%" actually lands on 30%, not "somewhere near 30%".
    """
    try:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        if action == "mute":
            volume.SetMute(1, None)
            return "Muted."
        if action == "unmute":
            volume.SetMute(0, None)
            return "Unmuted."
        if level is None:
            return "Please specify a volume level 0-100."
        level = max(0, min(100, level))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"Volume set to {level}%."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to change volume: {exc}"


