"""
System stats tool - CPU/RAM/GPU/VRAM, live from *this* machine.

why this tool is useful: a local voice assistant that
can tell you your own GPU temperature and VRAM headroom (on the very card
running its own LLM) is a much more convincing demo than a generic
"what's the weather" tool - it proves the agent has real system access,
not just canned API calls.
"""

from __future__ import annotations

import psutil

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_system_stats",
        "description": (
            "Get live system stats: CPU usage/temperature, RAM usage, and "
            "GPU utilization/VRAM/temperature. Use this whenever the user "
            "asks about their computer's performance, load, or hardware."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _gpu_stats() -> str:
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if not gpus:
            return "No GPU detected."
        lines = []
        for gpu in gpus:
            lines.append(
                f"{gpu.name}: {gpu.load * 100:.0f}% utilization, "
                f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB VRAM, "
                f"{gpu.temperature:.0f}C"
            )
        return " | ".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"GPU stats unavailable ({exc})"


def get_system_stats() -> str:
    cpu_pct = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    gpu_line = _gpu_stats()
    return (
        f"CPU: {cpu_pct:.0f}% utilization. "
        f"RAM: {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB used ({mem.percent:.0f}%). "
        f"GPU: {gpu_line}"
    )
