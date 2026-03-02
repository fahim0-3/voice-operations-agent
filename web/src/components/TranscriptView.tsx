/**
 * Scrolling conversation transcript - user turns (from STT) interleaved
 * with assistant turns (streamed token-by-token as they arrive, so text
 * appears progressively rather than popping in all at once).
 */

import { useEffect, useRef } from "react";
import type { TranscriptEntry } from "../hooks/useVoiceSocket";

export function TranscriptView({ entries }: { entries: TranscriptEntry[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <div className="transcript">
      {entries.length === 0 && (
        <div className="transcript__empty">Say something to begin the conversation.</div>
      )}
      {entries.map((entry, i) => (
        <div key={`${entry.turnId}-${entry.role}-${i}`} className={`transcript__entry transcript__entry--${entry.role}`}>
          <span className="transcript__role">{entry.role === "user" ? "You" : "Agent"}</span>
          <span className="transcript__text">{entry.text}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
