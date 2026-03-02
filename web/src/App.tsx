/// <reference types="vite/client" />
/**
 * Root component: wires the useVoiceSocket hook to the visual components
 * and exposes the single "Connect & Start Mic" control. Deliberately
 * minimal - the interesting engineering lives in the hook/audio modules,
 * not here.
 */

import { useCallback, useState } from "react";
import { LatencyHud } from "./components/LatencyHud";
import { StatusIndicator } from "./components/StatusIndicator";
import { TranscriptView } from "./components/TranscriptView";
import { useVoiceSocket } from "./hooks/useVoiceSocket";

const WS_URL =
  import.meta.env.VITE_WS_URL ??
  `${location.protocol === "https:" ? "wss" : "ws"}://127.0.0.1:8000/ws`;

export default function App() {
  const {
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
  } = useVoiceSocket(WS_URL);

  const [starting, setStarting] = useState(false);

  const handleStart = useCallback(async () => {
    setStarting(true);
    try {
      connect();
      await startMic();
    } finally {
      setStarting(false);
    }
  }, [connect, startMic]);

  return (
    <div className="app">
      <header className="app__header">
        <h1>Voice Operations Agent</h1>
        <p className="app__subtitle">A fully local, real-time voice agent.</p>
      </header>

      <StatusIndicator
        connectionState={connectionState}
        isUserSpeaking={isSpeaking}
        isAssistantSpeaking={isAssistantSpeaking}
        activeToolCall={activeToolCall}
      />

      <div className="app__controls">
        {!micActive ? (
          <button className="app__mic-button" onClick={handleStart} disabled={starting}>
            {starting ? "Connecting…" : "🎙 Start talking"}
          </button>
        ) : (
          <button className="app__mic-button app__mic-button--active" onClick={stopMic}>
            ⏹ Stop
          </button>
        )}
      </div>

      <main className="app__main">
        <TranscriptView entries={transcript} />
        <LatencyHud report={lastLatency} />
      </main>
    </div>
  );
}
