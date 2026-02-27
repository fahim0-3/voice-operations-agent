/**
 * TypeScript mirror of the server's app/protocol.py.
 *
 * why hand-maintain a duplicate type definition instead of generating it:
 * for a project this size, a codegen step (e.g. pydantic -> TS via
 * datamodel-codegen) would be more machinery than the problem warrants.
 * Keeping the two files side by side, with a comment pointing at the
 * Python source of truth, is a documented tradeoff - see
 * docs/11-websocket-protocol.md "keeping client/server types in sync".
 */

export interface ServerVadState {
  type: "vad_state";
  speaking: boolean;
}

export interface ServerPartialTranscript {
  type: "partial_transcript";
  text: string;
}

export interface ServerFinalTranscript {
  type: "final_transcript";
  text: string;
  turn_id: string;
}

export interface ServerLlmToken {
  type: "llm_token";
  turn_id: string;
  delta: string;
}

export interface ServerToolCall {
  type: "tool_call";
  turn_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
}

export interface ServerToolResult {
  type: "tool_result";
  turn_id: string;
  tool_name: string;
  result: string;
}

export interface ServerAudioStart {
  type: "audio_start";
  turn_id: string;
  sample_rate: number;
}

export interface ServerAudioEnd {
  type: "audio_end";
  turn_id: string;
}

export interface ServerBargeIn {
  type: "barge_in";
  interrupted_turn_id: string;
}

export interface ServerLatencyReport {
  type: "latency_report";
  turn_id: string;
  stages_ms: Record<string, number>;
  total_ms: number;
}

export interface ServerErrorMsg {
  type: "error";
  message: string;
}

export interface ServerTimerFired {
  type: "timer_fired";
  label: string;
}

export type ServerMessage =
  | ServerVadState
  | ServerPartialTranscript
  | ServerFinalTranscript
  | ServerLlmToken
  | ServerToolCall
  | ServerToolResult
  | ServerAudioStart
  | ServerAudioEnd
  | ServerBargeIn
  | ServerLatencyReport
  | ServerErrorMsg
  | ServerTimerFired;

export function parseServerMessage(raw: string): ServerMessage | null {
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.type === "string") {
      return parsed as ServerMessage;
    }
    return null;
  } catch {
    return null;
  }
}
