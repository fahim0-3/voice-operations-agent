# Voice Operations Agent

A local-first, real-time voice agent for a small guard-operations demo workflow. The browser streams microphone audio to a FastAPI backend over WebSockets. The backend uses Silero VAD to detect turns, Faster-Whisper for transcription, a deterministic safety/policy layer for known operational actions, and Piper TTS to return speech. A user can interrupt an in-progress reply (barge-in).

> [!IMPORTANT]
> This is a portfolio/demo application. Its operations tools write local demo records only. It does not contact guards, controllers, emergency services, or any production system.

## Features

- Real-time browser microphone capture over WebSockets.
- Silero VAD turn detection and barge-in cancellation.
- Local Faster-Whisper speech-to-text.
- Piper text-to-speech streamed to the browser as PCM audio.
- Pluggable LLM providers: local Ollama, AWS Bedrock, or an OpenAI-compatible API such as Groq.
- Deterministic handling for known guard-operations requests before an LLM is needed.
- Demo tools: welfare checks, guard-status lookups, shift-delay reports, incident records, and human-controller escalations.
- JSON Lines audit records stored locally.
- Per-turn latency measurements plus in-process p50, p95, and p99 summaries.

## Architecture

```text
Browser microphone
  -> WebSocket -> Silero VAD -> Faster-Whisper STT -> deterministic operations policy
                                                        -> local demo tool + concise spoken acknowledgement
                                                        -> human escalation for emergency or uncertain requests
                                                        -> LLM provider + approved tools for other requests
  -> Piper TTS -> WebSocket -> browser speaker

New user speech during a reply -> VAD -> cancel active LLM/TTS task -> stop browser audio
```

## Supported operations

| Spoken intent | Local demo action |
|---|---|
| “Log a welfare check for guard G12, status okay.” | `log_welfare_check` |
| “What is the status of guard G21?” | `get_guard_status` |
| “Guard G07 will be 15 minutes late.” | `report_shift_delay` |
| “Create an incident: suspicious person near gate 2.” | `create_incident` |
| “There is a fire at the site.” | `escalate_to_controller` and an emergency safety response |

Known operations are handled by transparent rules and recorded locally. Ambiguous operations requests and emergency keywords are escalated for human follow-up. The application never places a call or dispatches emergency services.

## Requirements

- Python 3.11 or 3.12
- Node.js 18 or newer
- [uv](https://docs.astral.sh/uv/)
- A Piper voice model: `en_US-lessac-medium.onnx` and its matching `.onnx.json` file in `server/assets/voices/`
- An LLM provider only for normal conversational or unmatched requests

An NVIDIA GPU is optional. The default STT configuration uses CUDA. For CPU-only execution, set `STT_DEVICE=cpu` and `STT_COMPUTE_TYPE=int8` in `server/.env`.

## Run locally

### Backend

```powershell
cd server
uv sync
Copy-Item .env.example .env
uv run uvicorn app.main:app --reload
```

### Configure an LLM provider

Select one provider in `server/.env`.

**Local Ollama**

```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

**Groq (OpenAI-compatible API)**

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_groq_api_key
OPENAI_MODEL=llama-3.1-8b-instant
OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

**AWS Bedrock**

```env
LLM_PROVIDER=bedrock
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
```

`LLM_PROVIDER=openai` selects the project's OpenAI-compatible connector. With the Groq base URL above, requests go to Groq, not OpenAI. Keep `.env` private; it is ignored by Git.

### Frontend

In a second terminal:

```powershell
cd web
npm install
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`). Allow microphone access, click **Start talking**, and speak a supported request.

## Verify

```powershell
cd server
uv run pytest -q

cd ..\web
npm run build
```

For a manual smoke test, use one supported operation at a time and wait for the Agent response. Then inspect health and aggregate latency:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/metrics/evaluation
```

`/metrics/evaluation` contains p50/p95/p99 values for completed turns. It reports STT, LLM time-to-first-token when an LLM is used, TTS first audio, and end-to-end completion. These figures are local measurements, not performance guarantees.

## Data and safety

- Audit data is stored locally at `server/data/operations-audit.jsonl` by default. Set `AUDIT_LOG_PATH` to change the location.
- The operational tools are local demonstrations, not integrations with a guard-management platform.
- Do not use this project as an emergency-response system. For immediate danger, contact local emergency services.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.