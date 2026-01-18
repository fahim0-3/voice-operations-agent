"""
Protocol schema tests.

why test the protocol at all: these pydantic models are the *contract*
between server and client. A silent breaking change here (e.g. renaming a
field) would fail at runtime in the browser console, not in Python - these
tests catch that at commit time instead.
"""

from __future__ import annotations

import json

from app.protocol import (
    ClientHello,
    ServerBargeIn,
    ServerFinalTranscript,
    ServerLatencyReport,
)


def test_client_hello_defaults():
    hello = ClientHello()
    assert hello.type == "hello"
    assert hello.sample_rate == 16000


def test_server_final_transcript_roundtrip():
    msg = ServerFinalTranscript(text="hello world", turn_id="abc123")
    dumped = msg.model_dump_json()
    parsed = json.loads(dumped)
    assert parsed["type"] == "final_transcript"
    assert parsed["text"] == "hello world"
    assert parsed["turn_id"] == "abc123"


def test_server_barge_in_has_correct_type():
    msg = ServerBargeIn(interrupted_turn_id="xyz")
    assert msg.type == "barge_in"
    assert msg.interrupted_turn_id == "xyz"


def test_latency_report_serializes_stage_dict():
    msg = ServerLatencyReport(turn_id="t1", stages_ms={"stt": 120.5, "llm_total": 300.0}, total_ms=420.5)
    data = json.loads(msg.model_dump_json())
    assert data["stages_ms"]["stt"] == 120.5
    assert data["total_ms"] == 420.5
