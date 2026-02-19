"""
Shared pytest fixtures.

why fixtures instead of module-level singletons: STT/TTS model loading is
slow (real disk I/O + CUDA context init) - `scope="session"` fixtures load
each model exactly once for the entire test run instead of once per test
function, which is the difference between a test suite that takes 5
seconds vs. 5 minutes.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tts():
    from app.tts import PiperTTS

    return PiperTTS(voice="en_US-lessac-medium", models_dir="./assets/voices")


@pytest.fixture(scope="session")
def vad():
    from app.vad import SileroVAD

    return SileroVAD(sample_rate=16000, threshold=0.5, min_silence_ms=500)


@pytest.fixture
def silence_frame() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)
