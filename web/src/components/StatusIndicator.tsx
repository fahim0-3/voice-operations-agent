/**
 * The single glanceable "what is the agent doing right now" widget:
 * idle / listening (VAD detected user speech) / thinking (tool call in
 * flight) / speaking (TTS audio playing). This is what makes barge-in
 * demo-able - a viewer can SEE "speaking" flip to "listening" the
 * instant they interrupt.
 */

import type { ConnectionState } from "../hooks/useVoiceSocket";

interface Props {
  connectionState: ConnectionState;
  isUserSpeaking: boolean;
  isAssistantSpeaking: boolean;
  activeToolCall: string | null;
}

export function StatusIndicator({
  connectionState,
  isUserSpeaking,
  isAssistantSpeaking,
  activeToolCall,
}: Props) {
  const label = deriveLabel({ connectionState, isUserSpeaking, isAssistantSpeaking, activeToolCall });

  return (
    <div className={`status-indicator status-indicator--${label.toLowerCase().replace(/\s/g, "-")}`}>
      <span className="status-indicator__dot" />
      <span className="status-indicator__label">{label}</span>
    </div>
  );
}

function deriveLabel(props: Props): string {
  if (props.connectionState !== "connected") return "Disconnected";
  if (props.activeToolCall) return `Using ${props.activeToolCall}`;
  if (props.isAssistantSpeaking) return "Speaking";
  if (props.isUserSpeaking) return "Listening";
  return "Idle";
}
