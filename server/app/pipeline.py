"""
The orchestrator: ties VAD -> STT -> LLM(+tools) -> TTS into one
per-connection turn loop, with barge-in cancellation.

This is the most important file in the project - see docs/02-architecture.md
for the full data-flow diagram and docs/10-barge-in-and-async.md for a deep
dive on the cancellation model implemented here.

THE CORE IDEA (barge-in):
Every assistant turn (LLM generation + TTS playback) runs inside a single
`asyncio.Task`. While that task is running, we keep feeding incoming mic
audio to the VAD in parallel. The instant the VAD detects new speech
starting, we call `task.cancel()` on the in-flight assistant turn and send
a `barge_in` message so the client immediately stops audio playback. This
is why the assistant-turn logic lives in its own cancellable coroutine
rather than being inlined into the main receive loop - you cannot cleanly
cancel "the rest of this straight-line function" any other way in asyncio.

why this file talks to `LLMProvider` (app/providers/) instead of a
concrete LLM client: Voice Operations Agent can run against Bedrock, Ollama, or
OpenAI depending on user config (see docs/16-decisions-adr.md) - this
orchestrator is written once and works unchanged against all three,
because every provider implements the same `chat_with_tools()` contract.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import structlog

from app.config import settings
from app.audit import write_audit_event
from app.metrics import LatencyEvaluation, TurnTimer
from app.operations import build_confirmed_operation, classify_operational_intent
from app.persona import SYSTEM_PROMPT
from app.protocol import (
    ServerAudioEnd,
    ServerAudioStart,
    ServerBargeIn,
    ServerFinalTranscript,
    ServerLatencyReport,
    ServerLlmToken,
    ServerToolCall,
    ServerToolResult,
    ServerVadState,
)
from app.providers.base import LLMProvider
from app.stt import WhisperSTT
from app.tools import TOOL_SCHEMAS, execute_tool
from app.tts import PiperTTS, split_into_sentences
from app.vad import SileroVAD, create_default_vad

logger = structlog.get_logger()

FRAME_SAMPLES = int(settings.audio_sample_rate * settings.audio_frame_ms / 1000)


class ConversationPipeline:
    """
    One instance of this class is created per WebSocket connection and
    owns all per-session state: the VAD's running turn-state, conversation
    history sent to the LLM, and whichever assistant-turn task is
    currently in flight (if any).
    """

    def __init__(
        self, send_json, send_bytes, stt: WhisperSTT, tts: PiperTTS, llm: LLMProvider,
        latency_evaluation: LatencyEvaluation | None = None,
    ):
        self.send_json = send_json  # async fn(pydantic model) -> None, provided by main.py
        self.send_bytes = send_bytes  # async fn(bytes) -> None
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.latency_evaluation = latency_evaluation

        self.vad: SileroVAD = create_default_vad()
        self.history: list[dict] = []
        self._speech_buffer: list[np.ndarray] = []
        self._current_turn_task: asyncio.Task | None = None
        self._current_turn_id: str | None = None

    # ------------------------------------------------------------------
    # Mic audio ingestion
    # ------------------------------------------------------------------

    async def handle_audio_frame(self, frame: np.ndarray) -> None:
        """
        Called once per incoming ~20ms audio frame from the client.

        why VAD runs unconditionally, even mid-assistant-turn: barge-in
        detection *requires* we keep listening while the assistant is
        talking - if we only ran VAD when idle, the user could never
        interrupt.
        """
        # The browser supplies Silero-compatible 512-sample frames at 16 kHz.
        # Derive timing from the received frame rather than assuming the old
        # 20 ms transport frame, so endpointing remains accurate.
        frame_ms = len(frame) / settings.audio_sample_rate * 1000.0
        is_speech, event = self.vad.process_frame(frame, frame_ms)
        await self.send_json(ServerVadState(speaking=is_speech))

        if event == "speech_start":
            self._speech_buffer = [frame]
            if self._current_turn_task and not self._current_turn_task.done():
                await self._barge_in()
        elif event == "speech_end":
            self._speech_buffer.append(frame)
            await self._on_user_turn_complete()
        elif is_speech:
            self._speech_buffer.append(frame)

    async def _barge_in(self) -> None:
        """
        Cancel the in-flight assistant turn and tell the client to stop
        audio playback immediately. See module docstring for why this
        works via asyncio.Task.cancel() rather than a manual flag.
        """
        assert self._current_turn_task is not None
        interrupted_id = self._current_turn_id or "unknown"
        self._current_turn_task.cancel()
        try:
            await self._current_turn_task
        except asyncio.CancelledError:
            pass  # expected - this is exactly what we asked for
        await self.send_json(ServerBargeIn(interrupted_turn_id=interrupted_id))
        logger.info("barge_in", interrupted_turn_id=interrupted_id)

    async def _on_user_turn_complete(self) -> None:
        if not self._speech_buffer:
            return
        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer = []
        turn_id = str(uuid.uuid4())
        self._current_turn_id = turn_id
        self._current_turn_task = asyncio.create_task(self._run_assistant_turn(audio, turn_id))

    # ------------------------------------------------------------------
    # The cancellable assistant turn: STT -> LLM(+tools) -> TTS
    # ------------------------------------------------------------------

    async def _run_assistant_turn(self, audio: np.ndarray, turn_id: str) -> None:
        timer = TurnTimer(turn_id=turn_id)

        transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        timer.mark("stt")
        if not transcript:
            return
        await self.send_json(ServerFinalTranscript(text=transcript, turn_id=turn_id))
        self.history.append({"role": "user", "content": transcript})

        decision = classify_operational_intent(transcript)
        write_audit_event(
            "policy_decision", turn_id=turn_id, intent=decision.intent,
            confidence=decision.confidence, requires_human_escalation=decision.requires_human_escalation,
            reason=decision.reason,
        )
        if decision.requires_human_escalation:
            result = execute_tool(
                "escalate_to_controller",
                {"reason": transcript, "severity": "emergency" if "emergency" in decision.reason else "uncertain"},
            )
            await self.send_json(ServerToolCall(
                turn_id=turn_id, tool_name="escalate_to_controller", arguments={"reason": transcript}
            ))
            await self.send_json(ServerToolResult(turn_id=turn_id, tool_name="escalate_to_controller", result=result))
            reply = (
                "This has been logged for immediate human-controller follow-up. "
                "If anyone is in immediate danger, contact local emergency services now."
                if "emergency" in decision.reason
                else "I could not safely classify that operations request. It has been logged for human-controller follow-up."
            )
            await self._speak_fixed_reply(reply, turn_id, timer)
            self.history.append({"role": "assistant", "content": reply})
            await self._finish_turn(timer, turn_id)
            return

        if decision.confidence == "high" and decision.intent != "unknown":
            tool_name, arguments, reply = build_confirmed_operation(transcript, decision.intent)
            result = execute_tool(tool_name, arguments)
            await self.send_json(ServerToolCall(turn_id=turn_id, tool_name=tool_name, arguments=arguments))
            await self.send_json(ServerToolResult(turn_id=turn_id, tool_name=tool_name, result=result))
            await self._speak_fixed_reply(reply, turn_id, timer)
            self.history.append({"role": "assistant", "content": reply})
            await self._finish_turn(timer, turn_id)
            return
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

        first_token = True

        async def on_token(delta: str) -> None:
            nonlocal first_token
            if first_token:
                timer.mark("llm_ttft")
                first_token = False
            await self.send_json(ServerLlmToken(turn_id=turn_id, delta=delta))

        async def on_tool_call(name: str, args: dict) -> None:
            await self.send_json(ServerToolCall(turn_id=turn_id, tool_name=name, arguments=args))

        async def on_tool_result(name: str, result: str) -> None:
            await self.send_json(ServerToolResult(turn_id=turn_id, tool_name=name, result=result))

        # Run TTS-consumer and LLM-producer concurrently so sentence N's
        # audio starts synthesizing while sentence N+1 is still being
        # generated - see docs/09-latency-engineering.md "pipelining".
        tts_task = asyncio.create_task(self._stream_tts(sentence_queue, turn_id, timer))

        full_reply = await self._stream_llm_into_sentences(sentence_queue, on_token, on_tool_call, on_tool_result)
        self.history.append({"role": "assistant", "content": full_reply})

        await tts_task
        await self._finish_turn(timer, turn_id)

    async def _speak_fixed_reply(self, reply: str, turn_id: str, timer: TurnTimer) -> None:
        """Start policy-owned replies with a short synthesized acknowledgement.

        Synthesizing the complete response before sending audio made the first
        spoken byte wait for Piper to finish all text. Processing one sentence
        at a time lets the client play the acknowledgement while remaining
        details are still being synthesized.
        """
        await self.send_json(ServerLlmToken(turn_id=turn_id, delta=reply))
        started = False
        for sentence in split_into_sentences(reply):
            pcm = await asyncio.to_thread(self.tts.synthesize_pcm, sentence)
            if not started:
                timer.mark("tts_first_audio")
                await self.send_json(ServerAudioStart(turn_id=turn_id, sample_rate=settings.tts_sample_rate))
                started = True
            if pcm:
                await self.send_bytes(pcm)
        await self.send_json(ServerAudioEnd(turn_id=turn_id))
    async def _finish_turn(self, timer: TurnTimer, turn_id: str) -> None:
        timer.mark_elapsed("end_to_end")
        total_ms = timer.total_ms
        if self.latency_evaluation:
            self.latency_evaluation.record(timer.stages_ms, total_ms)
        await self.send_json(ServerLatencyReport(turn_id=turn_id, stages_ms=timer.stages_ms, total_ms=total_ms))
        logger.info("turn_complete", **timer.summary())

    async def _stream_llm_into_sentences(self, sentence_queue, on_token, on_tool_call, on_tool_result) -> str:
        buffer = ""

        async def on_token_wrapped(delta: str) -> None:
            nonlocal buffer
            buffer += delta
            await on_token(delta)
            sentences = split_into_sentences(buffer)
            # emit all *complete* sentences immediately, keep any trailing
            # partial fragment in `buffer` for the next token
            if len(sentences) > 1 or (sentences and buffer.rstrip().endswith((".", "!", "?"))):
                complete = sentences if buffer.rstrip().endswith((".", "!", "?")) else sentences[:-1]
                for s in complete:
                    await sentence_queue.put(s)
                buffer = "" if buffer.rstrip().endswith((".", "!", "?")) else sentences[-1]

        reply = await self.llm.chat_with_tools(
            system_prompt=SYSTEM_PROMPT,
            history=self.history,
            tools=TOOL_SCHEMAS,
            on_token=on_token_wrapped,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
        if buffer.strip():
            await sentence_queue.put(buffer)
        await sentence_queue.put(None)  # sentinel: no more sentences coming
        return reply

    async def _stream_tts(self, sentence_queue: asyncio.Queue, turn_id: str, timer: TurnTimer) -> None:
        first_chunk = True
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            pcm = await asyncio.to_thread(self.tts.synthesize_pcm, sentence)
            if first_chunk:
                timer.mark("tts_first_audio")
                await self.send_json(
                    ServerAudioStart(turn_id=turn_id, sample_rate=settings.tts_sample_rate)
                )
                first_chunk = False
            await self.send_bytes(pcm)
        await self.send_json(ServerAudioEnd(turn_id=turn_id))
