"""Protect the browser-to-Silero audio frame contract."""

from __future__ import annotations

import numpy as np

from app.config import settings


def test_silero_transport_frame_is_32ms_at_16khz():
    """Silero VAD requires 512 samples when running at the configured 16 kHz."""
    frame = np.zeros(512, dtype=np.float32)
    frame_ms = len(frame) / settings.audio_sample_rate * 1000.0
    assert frame_ms == 32.0
