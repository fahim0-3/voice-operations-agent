"""
WebSocket wire protocol.

why a typed protocol at all: the browser and server exchange two very
different kinds of data over the *same* socket - raw binary PCM audio
frames (mic -> server, server -> speaker) and structured JSON events
(status updates, transcripts, tool calls). Rather than inventing an
ad-hoc scheme, every JSON message has a required `type` discriminator
field. Pydantic's discriminated unions let both sides validate messages
and give you autocomplete/type-checking instead of dict[str, Any] soup.

Binary frames are NOT wrapped in this protocol - see docs/11-websocket-protocol.md
for why binary audio bypasses JSON entirely (base64 would inflate it by ~33%
and add encode/decode latency we can't afford in a real-time path).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Client -> Server messages
# ---------------------------------------------------------------------------


class ClientHello(BaseModel):
    """Sent once, right after the socket opens, to configure the session."""

    type: Literal["hello"] = "hello"
    sample_rate: int = 16000
    wake_word_enabled: bool = False


class ClientTextMessage(BaseModel):
    """Lets the web UI send typed text instead of speaking (accessibility + debugging)."""

    type: Literal["text"] = "text"
    text: str


class ClientControl(BaseModel):
    """User-initiated control actions (e.g. mute mic, reset conversation)."""

    type: Literal["control"] = "control"
    action: Literal["mute", "unmute", "reset_conversation"]


ClientMessage = ClientHello | ClientTextMessage | ClientControl


# ---------------------------------------------------------------------------
# Server -> Client messages
# ---------------------------------------------------------------------------


class ServerVadState(BaseModel):
    """
    Fired whenever the VAD's speech/silence decision changes.

    why the client needs this: it drives the waveform UI (are we currently
    "hearing" you) and, combined with `turn_end`, tells the UI when to stop
    showing the mic-active indicator.
    """

    type: Literal["vad_state"] = "vad_state"
    speaking: bool


class ServerPartialTranscript(BaseModel):
    """Incremental STT output while the user is still speaking (optional/best-effort)."""

    type: Literal["partial_transcript"] = "partial_transcript"
    text: str


class ServerFinalTranscript(BaseModel):
    """The finalized transcript for one completed user turn."""

    type: Literal["final_transcript"] = "final_transcript"
    text: str
    turn_id: str


class ServerLlmToken(BaseModel):
    """One streamed token/chunk of the LLM's response, for live captions in the UI."""

    type: Literal["llm_token"] = "llm_token"
    turn_id: str
    delta: str


class ServerToolCall(BaseModel):
    """Announces that the agent is invoking a tool (shown in UI as e.g. "checking system stats...")."""

    type: Literal["tool_call"] = "tool_call"
    turn_id: str
    tool_name: str
    arguments: dict


class ServerToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    turn_id: str
    tool_name: str
    result: str


class ServerAudioStart(BaseModel):
    """Announces that binary TTS audio frames for this turn are about to start streaming."""

    type: Literal["audio_start"] = "audio_start"
    turn_id: str
    sample_rate: int


class ServerAudioEnd(BaseModel):
    """All TTS audio for this turn has been sent."""

    type: Literal["audio_end"] = "audio_end"
    turn_id: str


class ServerBargeIn(BaseModel):
    """
    Fired the instant new user speech is detected while the assistant is
    still talking. The client should immediately stop/flush audio playback.

    why this matters: this message is the entire point of the barge-in
    feature. Without an explicit signal, the client has no reliable way to
    know "stop everything, a new turn just started" - see docs/10-barge-in-and-async.md.
    """

    type: Literal["barge_in"] = "barge_in"
    interrupted_turn_id: str


class ServerLatencyReport(BaseModel):
    """Per-turn latency breakdown, shown in the UI's debug HUD."""

    type: Literal["latency_report"] = "latency_report"
    turn_id: str
    stages_ms: dict[str, float]
    total_ms: float


class ServerError(BaseModel):
    type: Literal["error"] = "error"
    message: str


ServerMessage = (
    ServerVadState
    | ServerPartialTranscript
    | ServerFinalTranscript
    | ServerLlmToken
    | ServerToolCall
    | ServerToolResult
    | ServerAudioStart
    | ServerAudioEnd
    | ServerBargeIn
    | ServerLatencyReport
    | ServerError
)


class Envelope(BaseModel):
    """
    why an envelope wrapper: pydantic discriminated unions need a common
    field to dispatch on when *parsing* incoming JSON of unknown shape.
    We parse into `Envelope` first to read `.type`, then re-validate into
    the concrete model. Outgoing messages skip this and just call
    `model_dump_json()` on the concrete model directly.
    """

    type: str
    payload: dict = Field(default_factory=dict)
