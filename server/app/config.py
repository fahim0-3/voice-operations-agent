"""
Centralized configuration.

why: every other module imports `settings` from here instead of calling
os.getenv() ad-hoc. That gives us one place to see every tunable knob,
one place to validate them (pydantic does this for free), and it makes
tests trivial (just construct a Settings object with overrides).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection ---
    # why a single switch instead of hardcoding one backend: the application
    # supports three interchangeable LLM backends (see app/providers/) so
    # the same voice pipeline can run fully offline (ollama), against a
    # managed multi-model service (bedrock - our default, since it gives
    # access to stronger models than anything that fits in 16GB VRAM
    # locally), or against OpenAI. Swapping is a one-line env var change,
    # no code edits - see docs/16-decisions-adr.md.
    llm_provider: str = "bedrock"  # "bedrock" | "ollama" | "openai"

    # --- AWS Bedrock ---
    bedrock_region: str = "us-east-1"
    # why claude-3-haiku (2024) instead of the newer claude-sonnet-5 /
    # claude-opus-5 / claude-haiku-4-5: this is a REAL-TIME VOICE pipeline,
    # where time-to-first-token (TTFT) directly becomes audible silence
    # before the agent starts speaking. Benchmarked on this project's own
    # account/region (see docs/16-decisions-adr.md for the raw numbers):
    #   claude-3-haiku-20240307   TTFT ~500ms   <- our default
    #   claude-haiku-4-5          TTFT ~1.7s
    #   claude-opus-5             TTFT ~1.3s
    #   claude-sonnet-5           TTFT ~1.7s
    #   claude-sonnet-4-5         TTFT ~1.9s
    # The newer models are smarter but 2.5-4x slower to first token - not
    # worth it for a voice agent's tool-routing/small-talk workload. If
    # you want maximum reasoning quality over latency (e.g. for a
    # text-only session, or if you don't mind the pause), swap to any
    # "us.anthropic.claude-..." inference-profile ID below - no code
    # changes needed, just change this one setting.
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    # AWS credentials are intentionally NOT settings here - boto3 resolves
    # them itself from ~/.aws/credentials, environment variables, or an
    # IAM role, in that standard order. Duplicating that logic here would
    # be redundant and risks accidentally logging a secret.

    # --- Ollama (fully local) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"


    # --- OpenAI (or any OpenAI-compatible endpoint) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""  # override to point at Groq/Together/local llama.cpp, etc.

    llm_max_history_turns: int = 12  # how many past turns we keep in context

    # --- STT ---
    stt_model: str = "distil-small.en"
    stt_device: str = "cuda"  # "cuda" or "cpu" - falls back automatically if cuda unavailable
    stt_compute_type: str = "float16"  # float16 on GPU, int8 on CPU

    # --- VAD (voice activity detection / turn-taking) ---
    vad_threshold: float = 0.5  # Silero VAD speech-probability threshold
    vad_min_silence_ms: int = 500  # silence duration that ends a turn
    vad_speech_pad_ms: int = 200  # pad kept around detected speech

    # --- TTS ---
    tts_voice: str = "en_US-lessac-medium"
    tts_models_dir: str = "./assets/voices"
    tts_sample_rate: int = 22050  # native rate of en_US-lessac-medium; read tts.sample_rate at runtime for other voices

    # --- Audio ---
    audio_sample_rate: int = 16000  # what the mic/STT pipeline runs at
    audio_frame_ms: int = 20  # size of each audio frame sent over the wire

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- Agent persona ---
    agent_name: str = "Voice Operations Agent"
    wake_word_enabled: bool = False

    # --- Local operational audit ---
    audit_log_path: str = "./data/operations-audit.jsonl"


settings = Settings()
