"""
Per-turn latency instrumentation.

why this exists as its own module: "make it fast" is a vague goal you can't
verify. "Get p50 mic-to-first-audio-byte under 800ms" is a goal you can
prove with a number. This module is what turns our latency claims from
marketing copy into something a skeptical reviewer can check in
bench/results.md.

Design: a `TurnTimer` is created once per conversational turn. Each pipeline
stage calls `.mark(stage_name)` when it finishes. At the end we get a
dict of stage -> duration_ms plus a total, which we both ship to the client
(for the live HUD) and log (for the benchmark harness to aggregate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import quantiles


@dataclass
class TurnTimer:
    turn_id: str
    _t0: float = field(default_factory=time.perf_counter)
    _last: float = field(default_factory=time.perf_counter)
    stages_ms: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> float:
        """
        Record the elapsed time (ms) since the *previous* mark (or start)
        under `stage`. Returns that duration so callers can log it inline.

        why relative-to-previous instead of relative-to-start: it lets us
        see the cost of *each* pipeline stage independently (VAD, STT, LLM
        TTFT, TTS-first-chunk) rather than only a cumulative number - that's
        what tells us *where* to optimize.
        """
        now = time.perf_counter()
        dt_ms = (now - self._last) * 1000.0
        self.stages_ms[stage] = round(dt_ms, 2)
        self._last = now
        return dt_ms

    @property
    def total_ms(self) -> float:
        return round((self._last - self._t0) * 1000.0, 2)

    def summary(self) -> dict:
        return {"turn_id": self.turn_id, "stages_ms": self.stages_ms, "total_ms": self.total_ms}

    def mark_since(self, stage: str, earlier_stage: str | None = None) -> float:
        """Store a named duration since start or a previous stage without changing legacy marks."""
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        if earlier_stage and earlier_stage in self.stages_ms:
            elapsed -= self.stages_ms[earlier_stage]
        self.stages_ms[stage] = round(elapsed, 2)
        return elapsed

    def mark_elapsed(self, stage: str) -> float:
        """Record elapsed wall time from turn start (for end-to-end latency)."""
        now = time.perf_counter()
        elapsed = (now - self._t0) * 1000.0
        self.stages_ms[stage] = round(elapsed, 2)
        self._last = now
        return elapsed


@dataclass
class LatencyEvaluation:
    """In-memory percentile summaries for the current server process."""

    reports: list[dict[str, float]] = field(default_factory=list)
    max_reports: int = 500

    def record(self, stages_ms: dict[str, float], total_ms: float) -> None:
        self.reports.append({**stages_ms, "end_to_end": total_ms})
        del self.reports[:-self.max_reports]

    def summary(self) -> dict:
        keys = sorted({key for report in self.reports for key in report})
        measurements: dict[str, dict[str, float]] = {}
        for key in keys:
            values = sorted(report[key] for report in self.reports if key in report)
            if not values:
                continue
            measurements[key] = {
                "count": len(values),
                "p50": round(_percentile(values, 50), 2),
                "p95": round(_percentile(values, 95), 2),
                "p99": round(_percentile(values, 99), 2),
            }
        return {"turns": len(self.reports), "metrics_ms": measurements}


def _percentile(values: list[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[percentile - 1]
