"""
Latency benchmark harness.

why a standalone script rather than folding this into pytest: benchmarks
are not correctness tests - they're meant to be run manually, produce
human-readable numbers, and be re-run whenever you change hardware/models
to update the README's benchmark table. Mixing that into the pytest suite
(which should be fast and always-green) would be the wrong tool for the
job - see docs/14-testing-and-ci.md.

Usage:
    cd server
    uv run python ../bench/latency_bench.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))


def bench_stt(n_runs: int = 5) -> None:
    from app.stt import WhisperSTT
    import numpy as np

    print("\n--- STT (faster-whisper) ---")
    stt = WhisperSTT()
    audio = (np.random.randn(16000 * 3) * 0.01).astype(np.float32)  # 3s

    stt.transcribe(audio)  # warmup (compiles kernels)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        stt.transcribe(audio)
        times.append(time.perf_counter() - t0)

    _report("STT (3s audio)", times, audio_duration_s=3.0)


def bench_tts(n_runs: int = 5) -> None:
    from app.tts import PiperTTS

    print("\n--- TTS (Piper) ---")
    tts = PiperTTS(voice="en_US-lessac-medium", models_dir="./assets/voices")
    text = "The quick brown fox jumps over the lazy dog near the riverbank."

    tts.synthesize_pcm(text)  # warmup

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        tts.synthesize_pcm(text)
        times.append(time.perf_counter() - t0)

    _report("TTS (1 sentence)", times)


def bench_vad(n_frames: int = 200) -> None:
    import numpy as np
    from app.vad import SileroVAD

    print("\n--- VAD (Silero) per-frame cost ---")
    vad = SileroVAD()
    frame = (np.random.randn(512) * 0.1).astype(np.float32)

    vad.process_frame(frame, 32)  # warmup

    t0 = time.perf_counter()
    for _ in range(n_frames):
        vad.process_frame(frame, 32)
    total = time.perf_counter() - t0
    print(f"  {n_frames} frames in {total * 1000:.1f}ms -> {total / n_frames * 1000:.3f}ms/frame")


def bench_bedrock_ttft(model_ids: list[str]) -> None:
    import boto3

    print("\n--- Bedrock streaming TTFT ---")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    for model_id in model_ids:
        try:
            t0 = time.perf_counter()
            resp = client.converse_stream(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "Say hello in one short sentence."}]}],
                inferenceConfig={"maxTokens": 60},
            )
            ttft = None
            for event in resp["stream"]:
                if "contentBlockDelta" in event and ttft is None:
                    ttft = time.perf_counter() - t0
                    break
            print(f"  {model_id}: TTFT={ttft * 1000:.0f}ms")
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_id}: FAILED ({type(exc).__name__}: {str(exc)[:100]})")


def _report(label: str, times: list[float], audio_duration_s: float | None = None) -> None:
    mean_ms = statistics.mean(times) * 1000
    median_ms = statistics.median(times) * 1000
    print(f"  {label}: mean={mean_ms:.1f}ms median={median_ms:.1f}ms (n={len(times)})")
    if audio_duration_s:
        rtf = statistics.mean(times) / audio_duration_s
        print(f"    RTF = {rtf:.4f} (lower is faster-than-real-time)")


if __name__ == "__main__":
    print("Voice Operations Agent latency benchmark")
    print("=" * 40)
    bench_vad()
    bench_tts()
    bench_stt()
    bench_bedrock_ttft(
        [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        ]
    )
    print("\nDone. Copy relevant numbers into README.md's benchmark table.")
