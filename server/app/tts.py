"""
Text-to-Speech backend - Piper (VITS-family) via ONNX Runtime, on CPU.

why CPU instead of GPU for TTS: Piper's ONNX models are tiny and fast
enough that CPU inference for a sentence-length chunk takes tens of ms on
a modern desktop CPU - well within budget. Keeping TTS on CPU means the
*entire* 16GB of VRAM stays free for STT + the 7B LLM, which matters a lot
more for total system throughput than shaving a few ms off TTS. See
docs/08-tts.md and docs/15-gpu-cuda-and-vram.md for the VRAM budget math.

why sentence-chunked synthesis instead of waiting for the full LLM
response: synthesizing per-sentence lets audio *start playing* while the
LLM is still generating the rest of the reply - this is one of the two
biggest levers on perceived end-to-end latency (the other is TTFT). See
docs/09-latency-engineering.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np

from app.config import settings

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """
    why regex instead of a full sentence-tokenizer library (nltk/spacy):
    LLM output for a voice assistant is short, simple, conversational
    English - a heavyweight NLP tokenizer is overkill. This handles the
    99% case (. ! ?) and we accept minor edge-case imperfections
    (e.g. "Mr. Smith") as a documented tradeoff.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class PiperTTS:
    def __init__(self, voice: str | None = None, models_dir: str | None = None) -> None:
        from piper import PiperVoice  # imported lazily - heavy onnxruntime import

        voice_name = voice or settings.tts_voice
        models_dir = models_dir or settings.tts_models_dir
        model_path = f"{models_dir}/{voice_name}.onnx"
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize_pcm(self, text: str) -> bytes:
        """
        Returns 16-bit signed mono PCM bytes (no WAV header) at the voice's
        native sample rate - ready to be sent straight over the WebSocket
        as a binary frame. See docs/11-websocket-protocol.md for why we
        skip WAV framing entirely rather than sending a full .wav per
        sentence.

        why concatenate chunks here: Piper's synthesize() yields one
        AudioChunk *per sentence it detects internally* - for a single
        short sentence that's almost always exactly one chunk, but we
        concatenate defensively in case Piper's own sentence splitter
        disagrees with ours on a given input.
        """
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return b""
        audio = np.concatenate([c.audio_float_array for c in chunks])
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def synthesize_streaming(self, text_chunks: Iterator[str]) -> Iterator[bytes]:
        """
        Consumes an iterator of text chunks (e.g. sentences as they
        complete) and yields PCM audio bytes per chunk. This is the
        generator pipeline.py drives during a turn, so audio for sentence 1
        can be sent while the LLM is still producing sentence 3.
        """
        for chunk in text_chunks:
            if chunk.strip():
                yield self.synthesize_pcm(chunk)
