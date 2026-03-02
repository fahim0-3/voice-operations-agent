/**
 * Live latency breakdown overlay - the showpiece of the "latency
 * engineering" story this project tells. Shown after every turn so a
 * viewer (or you, benchmarking) can see exactly where time went:
 * STT vs LLM vs TTS-first-chunk, not just a single opaque total.
 */

import type { LatencyReport } from "../hooks/useVoiceSocket";

const STAGE_LABELS: Record<string, string> = {
  stt: "Speech-to-text",
  llm_ttft: "LLM first token",
  tts_first_audio: "TTS first audio",
  end_to_end: "End-to-end completion",
};

export function LatencyHud({ report }: { report: LatencyReport | null }) {
  if (!report) {
    return (
      <div className="latency-hud latency-hud--empty">
        <span>No turns yet — speak to see the latency breakdown.</span>
      </div>
    );
  }

  const stages = Object.entries(report.stagesMs);
  const maxMs = Math.max(...stages.map(([, ms]) => ms), 1);

  return (
    <div className="latency-hud">
      <div className="latency-hud__header">
        <strong>Turn latency</strong>
        <span className="latency-hud__total">{report.totalMs.toFixed(0)} ms total</span>
      </div>
      {stages.map(([stage, ms]) => (
        <div key={stage} className="latency-hud__row">
          <span className="latency-hud__label">{STAGE_LABELS[stage] ?? stage}</span>
          <div className="latency-hud__bar-track">
            <div
              className="latency-hud__bar-fill"
              style={{ width: `${(ms / maxMs) * 100}%` }}
            />
          </div>
          <span className="latency-hud__value">{ms.toFixed(0)} ms</span>
        </div>
      ))}
    </div>
  );
}
