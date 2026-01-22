"""
Voice Activity Detection (VAD) - turn-taking / endpointing.

why VAD instead of "just always be listening and transcribing": running
Whisper continuously on silence wastes GPU cycles and produces garbage
transcripts (it hallucinates text on silence/noise). VAD is a cheap
(~1ms per frame) binary classifier that answers "is there speech in this
20ms frame?" so we only spend expensive STT compute on actual speech, and
so we know *when a turn ends* (silence after speech = user is done talking).

We use Silero VAD (via the `silero-vad` PyPI package, which ships the
JIT-traced model directly - no torch.hub network fetch required, and no
torchaudio import at load time). It's a small model, not a full neural net
comparable to Whisper - the industry-standard cheap VAD.


See docs/04-vad-and-turn-taking.md for the full explanation of the
accuracy-vs-latency tradeoff in `min_silence_ms`.
"""

from __future__ import annotations

import numpy as np
import torch

from app.config import settings


class TurnState:
    """Tracks whether we're currently inside a user speech turn."""

    IDLE = "idle"
    SPEAKING = "speaking"
    TRAILING_SILENCE = "trailing_silence"  # speech ended, waiting to confirm turn-end


class SileroVAD:
    """
    Thin wrapper around Silero VAD that turns a stream of small audio frames
    into higher-level turn-taking events (speech_start, speech_end).

    why a stateful class instead of a pure function: VAD models are
    frame-by-frame, but turn detection needs *memory* across frames (how
    long has it been silent?). That state has to live somewhere - here.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms

        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=False)
        self.model.eval()

        self.state = TurnState.IDLE
        self._silence_ms_accum = 0.0


    @torch.inference_mode()
    def _speech_prob(self, frame: np.ndarray) -> float:
        """Silero expects float32 tensors in [-1, 1] at a fixed chunk size (e.g. 512 samples @16k)."""
        tensor = torch.from_numpy(frame).float()
        return float(self.model(tensor, self.sample_rate).item())

    def process_frame(self, frame: np.ndarray, frame_ms: float) -> tuple[bool, str | None]:
        """
        Feed one audio frame in. Returns (is_speech, event) where event is
        one of None / "speech_start" / "speech_end".

        why return an explicit event instead of making the caller diff
        state: the pipeline orchestrator needs a clean trigger to say
        "user just finished talking, kick off STT" - polling raw
        probabilities for that is error-prone and duplicated logic.
        """
        prob = self._speech_prob(frame)
        is_speech = prob >= self.threshold
        event: str | None = None

        if self.state == TurnState.IDLE:
            if is_speech:
                self.state = TurnState.SPEAKING
                event = "speech_start"

        elif self.state == TurnState.SPEAKING:
            if not is_speech:
                self.state = TurnState.TRAILING_SILENCE
                self._silence_ms_accum = frame_ms

        elif self.state == TurnState.TRAILING_SILENCE:
            if is_speech:
                # false alarm - speech resumed before the silence threshold elapsed
                self.state = TurnState.SPEAKING
                self._silence_ms_accum = 0.0
            else:
                self._silence_ms_accum += frame_ms
                if self._silence_ms_accum >= self.min_silence_ms:
                    self.state = TurnState.IDLE
                    self._silence_ms_accum = 0.0
                    event = "speech_end"

        return is_speech, event

    def reset(self) -> None:
        self.state = TurnState.IDLE
        self._silence_ms_accum = 0.0


def create_default_vad() -> SileroVAD:
    return SileroVAD(
        sample_rate=settings.audio_sample_rate,
        threshold=settings.vad_threshold,
        min_silence_ms=settings.vad_min_silence_ms,
    )
