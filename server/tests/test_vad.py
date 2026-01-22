"""
VAD turn-detection tests, using real TTS-synthesized speech (not noise -
Silero is a trained classifier and correctly ignores white noise, so
using noise as fake "speech" in a test would be testing nothing).
"""

from __future__ import annotations

import numpy as np


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    n_target = int(len(audio) * dst_sr / src_sr)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_target)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def test_vad_detects_speech_start_and_end_on_real_speech(vad, tts):
    pcm = tts.synthesize_pcm("Testing the voice activity detector with real speech audio.")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio_16k = _resample_linear(audio, tts.sample_rate, 16000)
    padded = np.concatenate(
        [np.zeros(16000, dtype=np.float32), audio_16k, np.zeros(16000, dtype=np.float32)]
    )

    vad.reset()
    frame_size = 512
    events = []
    for i in range(len(padded) // frame_size):
        frame = padded[i * frame_size : (i + 1) * frame_size]
        _is_speech, event = vad.process_frame(frame, 32)
        if event:
            events.append(event)

    assert events == ["speech_start", "speech_end"]


def test_vad_stays_idle_on_pure_silence(vad, silence_frame):
    vad.reset()
    for _ in range(50):
        is_speech, event = vad.process_frame(silence_frame, 32)
        assert not is_speech
        assert event is None
    assert vad.state == "idle"


def test_vad_reset_clears_state(vad):
    vad.state = "speaking"
    vad.reset()
    assert vad.state == "idle"
