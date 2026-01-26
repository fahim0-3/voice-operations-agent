"""
Speech-to-Text backend.

why faster-whisper (CTranslate2) instead of vanilla openai-whisper via
transformers: CT2 is a purpose-built inference engine (not a training
framework pressed into inference duty) - it gives ~4x faster inference
and lower VRAM via better kernel fusion and int8/float16 quantization.
We validated at project start that CT2 has working CUDA kernels for our
Blackwell (sm_120) GPU - see docs/15-gpu-cuda-and-vram.md for the spike
results (warm RTF ~0.02-0.07, i.e. transcribing 10s of audio in ~165ms).

We use "distil-small.en" - a distilled Whisper checkpoint that's ~6x
faster than the full model with a small accuracy tradeoff. For a
real-time voice agent, latency matters more than perfect WER on an
English-only assistant - see docs/05-stt-whisper.md.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings


class WhisperSTT:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model = WhisperModel(
            model_name or settings.stt_model,
            device=device or settings.stt_device,
            compute_type=compute_type or settings.stt_compute_type,
        )

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        audio: float32 mono PCM at 16kHz, values in [-1, 1].

        why beam_size=1 (greedy) by default here rather than faster-whisper's
        default beam_size=5: for short conversational utterances, greedy
        decoding is ~3-5x faster with a barely-noticeable accuracy cost -
        we're optimizing for the "sub-second" latency budget, not
        transcript-perfection (that tradeoff is measured in bench/).
        """
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,  # we run our own Silero VAD upstream - avoid double-gating
        )
        return "".join(seg.text for seg in segments).strip()
