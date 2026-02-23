"""
TurnTimer tests - verifying our latency measurement tool is itself accurate.

why this matters more than it might seem: every latency number we publish
in bench/results.md is only as trustworthy as this class. A bug here would
silently invalidate every benchmark claim in the README.
"""

from __future__ import annotations

import time

from app.metrics import LatencyEvaluation, TurnTimer


def test_mark_records_positive_duration():
    timer = TurnTimer(turn_id="t1")
    time.sleep(0.01)
    dt = timer.mark("stage_a")
    assert dt > 0


def test_marks_are_relative_not_cumulative():
    """
    why this is the important behavior: stages_ms['stage_b'] should be the
    time *since stage_a finished*, not the time since the timer started -
    that's what lets us see per-stage cost independently (see metrics.py
    docstring).
    """
    timer = TurnTimer(turn_id="t1")
    time.sleep(0.02)
    timer.mark("stage_a")
    time.sleep(0.05)
    timer.mark("stage_b")

    assert timer.stages_ms["stage_a"] < timer.stages_ms["stage_b"]


def test_total_ms_is_sum_of_all_marks_approximately():
    timer = TurnTimer(turn_id="t1")
    time.sleep(0.01)
    timer.mark("a")
    time.sleep(0.01)
    timer.mark("b")

    total_of_marks = sum(timer.stages_ms.values())
    assert abs(timer.total_ms - total_of_marks) < 1.0  # within 1ms rounding


def test_summary_shape():
    timer = TurnTimer(turn_id="turn-42")
    timer.mark("only_stage")
    summary = timer.summary()
    assert summary["turn_id"] == "turn-42"
    assert "only_stage" in summary["stages_ms"]
    assert "total_ms" in summary


def test_latency_evaluation_reports_percentiles():
    evaluation = LatencyEvaluation()
    for latency in [10.0, 20.0, 30.0]:
        evaluation.record({"stt": latency}, latency + 5)

    summary = evaluation.summary()
    assert summary["turns"] == 3
    assert summary["metrics_ms"]["stt"]["p50"] == 20.0
    assert summary["metrics_ms"]["end_to_end"]["p95"] >= 30.0
