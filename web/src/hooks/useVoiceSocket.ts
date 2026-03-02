/**
 * The central client-side state machine: owns the WebSocket connection,
 * mic capture, TTS playback, and the reactive UI state derived from
 * server events. Every other component just reads state from this hook.
 *
 * why one big hook instead of splitting socket/audio/state into three:
 * they're tightly coupled by nature (a `barge_in` message must both
 * update UI state AND call audioPlayer.stopImmediately() in the same
 * instant) - splitting them would just relocate the coupling into extra
 * prop-drilling without removing it. See docs/12-frontend-audio.md.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { startMicCapture, type MicCapture } from "../audio/capture";
import { AudioPlayer } from "../audio/playback";
import { parseServerMessage } from "../protocol";

export interface TranscriptEntry {
  turnId: string;
  role: "user" | "assistant";
  text: string;
}

export interface LatencyReport {
  turnId: string;
  stagesMs: Record<string, number>;
  totalMs: number;
}

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

export function useVoiceSocket(wsUrl: string) {
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [micActive, setMicActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false); // VAD: is the USER currently talking
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [activeToolCall, setActiveToolCall] = useState<string | null>(null);
  const [lastLatency, setLastLatency] = useState<LatencyReport | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const currentAssistantTextRef = useRef<string>("");
  const currentTurnIdRef = useRef<string>("");

  const connect = useCallback(() => {
    setConnectionState("connecting");
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setConnectionState("connected");
    ws.onerror = () => setConnectionState("error");
    ws.onclose = () => setConnectionState("idle");

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // binary frame = TTS PCM audio for the currently-speaking turn
        playerRef.current?.enqueuePcm16(event.data);
        return;
      }

      const msg = parseServerMessage(event.data as string);
      if (!msg) return;

      switch (msg.type) {
        case "vad_state":
          setIsSpeaking(msg.speaking);
          break;

        case "final_transcript":
          setTranscript((prev) => [...prev, { turnId: msg.turn_id, role: "user", text: msg.text }]);
          break;

        case "llm_token":
          currentTurnIdRef.current = msg.turn_id;
          currentAssistantTextRef.current += msg.delta;
          setTranscript((prev) => upsertAssistantEntry(prev, msg.turn_id, currentAssistantTextRef.current));
          break;

        case "tool_call":
          setActiveToolCall(msg.tool_name);
          break;

        case "tool_result":
          setActiveToolCall(null);
          break;

        case "audio_start":
          setIsAssistantSpeaking(true);
          currentAssistantTextRef.current = "";
          break;

        case "audio_end":
          setIsAssistantSpeaking(false);
          break;

        case "barge_in":
          // the whole point of barge-in: kill audio playback the INSTANT
          // this arrives, don't wait for React to re-render first.
          playerRef.current?.stopImmediately();
          setIsAssistantSpeaking(false);
          break;

        case "latency_report":
          setLastLatency({ turnId: msg.turn_id, stagesMs: msg.stages_ms, totalMs: msg.total_ms });
          break;
      }
    };

    wsRef.current = ws;
  }, [wsUrl]);

  const startMic = useCallback(async () => {
    if (!playerRef.current) {
      playerRef.current = new AudioPlayer(22050); // matches server's tts_sample_rate default
    }
    const capture = await startMicCapture((pcm16) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(pcm16);
      }
    });
    micRef.current = capture;
    setMicActive(true);
  }, []);

  const stopMic = useCallback(() => {
    micRef.current?.stop();
    micRef.current = null;
    setMicActive(false);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      micRef.current?.stop();
      playerRef.current?.close();
    };
  }, []);

  return {
    connectionState,
    micActive,
    isSpeaking,
    isAssistantSpeaking,
    transcript,
    activeToolCall,
    lastLatency,
    connect,
    startMic,
    stopMic,
  };
}

function upsertAssistantEntry(
  prev: TranscriptEntry[],
  turnId: string,
  text: string,
): TranscriptEntry[] {
  const idx = prev.findIndex((e) => e.turnId === turnId && e.role === "assistant");
  if (idx === -1) {
    return [...prev, { turnId, role: "assistant", text }];
  }
  const next = [...prev];
  next[idx] = { ...next[idx], text };
  return next;
}
