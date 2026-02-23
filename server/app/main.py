"""
FastAPI application entrypoint: HTTP health check + the /ws WebSocket
endpoint that drives one ConversationPipeline per connected client.

why WebSocket instead of REST/SSE for this: a voice agent is inherently
bidirectional and continuous - the client is *streaming* mic audio in
while *simultaneously* receiving transcript/LLM-token/TTS-audio events
out. REST req/response can't model that; SSE is one-way. WebSocket is the
only browser-native transport that supports low-latency full-duplex
binary + text framing - see docs/11-websocket-protocol.md.
"""

from __future__ import annotations

import struct

import numpy as np
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.metrics import LatencyEvaluation
from app.pipeline import ConversationPipeline
from app.providers import create_provider
from app.providers.base import LLMProvider
from app.stt import WhisperSTT
from app.tools import timer as timer_tool
from app.tts import PiperTTS


logger = structlog.get_logger()

app = FastAPI(title="Voice Operations Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are loaded once at process startup (not per-connection) - loading
# a Whisper/Piper model takes real time (disk + CUDA context init) and
# they are stateless/thread-safe enough to share across connections.
_stt: WhisperSTT | None = None
_tts: PiperTTS | None = None
_llm: LLMProvider | None = None
latency_evaluation = LatencyEvaluation()


@app.on_event("startup")
async def load_models() -> None:
    global _stt, _tts, _llm
    logger.info("loading_models", stt_model=settings.stt_model, llm_provider=settings.llm_provider)
    _stt = WhisperSTT()
    _tts = PiperTTS()
    # why create_provider() with no args: it reads settings.llm_provider
    # (bedrock by default) - swapping backends is a .env change, not a
    # code change. See app/providers/__init__.py.
    _llm = create_provider()
    logger.info("models_loaded", llm_provider=_llm.name)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "stt_loaded": _stt is not None,
        "tts_loaded": _tts is not None,
        "llm_provider": _llm.name if _llm else None,
    }


@app.get("/providers")
async def providers() -> dict:
    """
    Lets the web UI show/select available LLM providers without hardcoding
    the list client-side - see docs/16-decisions-adr.md for the
    multi-provider design.
    """
    from app.providers import VALID_PROVIDERS

    return {"active": settings.llm_provider, "available": list(VALID_PROVIDERS)}


@app.get("/metrics/evaluation")
async def metrics_evaluation() -> dict:
    """p50/p95/p99 for completed turns in this server process (no telemetry export)."""
    return latency_evaluation.summary()



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    assert _stt and _tts and _llm  # startup event guarantees these are loaded

    async def send_json(model) -> None:
        await websocket.send_text(model.model_dump_json())

    async def send_bytes(data: bytes) -> None:
        await websocket.send_bytes(data)

    pipeline = ConversationPipeline(
        send_json=send_json, send_bytes=send_bytes, stt=_stt, tts=_tts, llm=_llm,
        latency_evaluation=latency_evaluation,
    )

    # timers need a way to push a message back to *this* connection
    async def notify_timer(label: str) -> None:
        await websocket.send_text(f'{{"type":"timer_fired","label":"{label}"}}')

    timer_tool.register_notifier(notify_timer)

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                frame = _bytes_to_frame(message["bytes"])
                await pipeline.handle_audio_frame(frame)
            elif "text" in message and message["text"] is not None:
                # JSON control messages (hello/text/control) - see app/protocol.py
                logger.debug("text_message_received", raw=message["text"])
    except WebSocketDisconnect:
        logger.info("client_disconnected")


def _bytes_to_frame(data: bytes) -> np.ndarray:
    """
    Incoming binary frames are 16-bit signed PCM (what the browser's
    AudioWorklet sends - see web/src/audio/worklet.ts). Whisper/Silero
    expect float32 in [-1, 1], so we convert here once at the boundary.
    """
    count = len(data) // 2
    ints = struct.unpack(f"<{count}h", data)
    return np.array(ints, dtype=np.float32) / 32768.0
