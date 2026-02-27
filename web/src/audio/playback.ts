/**
 * TTS audio playback with an internal jitter buffer + hard-stop support
 * for barge-in.
 *
 * why a custom queue instead of just creating an <audio> element per
 * chunk: PCM chunks arrive continuously as the LLM streams sentences: we
 * need seamless back-to-back playback with zero gaps between chunks, and
 * we need `stop()` to be instant and total (every scheduled buffer
 * cancelled) the moment a barge_in event arrives - see
 * docs/10-barge-in-and-async.md for why "instant" is the whole point.
 */

export class AudioPlayer {
  private audioContext: AudioContext;
  private nextStartTime = 0;
  private activeSources: AudioBufferSourceNode[] = [];
  private sampleRate: number;

  constructor(sampleRate: number) {
    this.sampleRate = sampleRate;
    this.audioContext = new AudioContext({ sampleRate });
  }

  /** Enqueue a raw 16-bit PCM chunk for gapless playback. */
  enqueuePcm16(pcm: ArrayBuffer): void {
    const float32 = pcm16ToFloat32(pcm);
    const audioBuffer = this.audioContext.createBuffer(1, float32.length, this.sampleRate);
    // why getChannelData().set() instead of copyToChannel(): some TS lib
    // versions type copyToChannel's parameter as requiring a plain
    // ArrayBuffer-backed Float32Array (rejecting the more general
    // ArrayBufferLike our decode function returns) - .set() has no such
    // restriction and does the identical copy.
    audioBuffer.getChannelData(0).set(float32);


    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const now = this.audioContext.currentTime;
    const startTime = Math.max(now, this.nextStartTime);
    source.start(startTime);
    this.nextStartTime = startTime + audioBuffer.duration;

    this.activeSources.push(source);
    source.onended = () => {
      this.activeSources = this.activeSources.filter((s) => s !== source);
    };
  }

  /**
   * Hard-stop everything immediately - called the instant a `barge_in`
   * message arrives. This is the browser-side half of the barge-in
   * feature; app/pipeline.py's `_barge_in()` is the server-side half.
   */
  stopImmediately(): void {
    for (const source of this.activeSources) {
      try {
        source.stop();
      } catch {
        // already stopped/ended - safe to ignore
      }
    }
    this.activeSources = [];
    this.nextStartTime = this.audioContext.currentTime;
  }

  close(): void {
    this.stopImmediately();
    this.audioContext.close();
  }
}

function pcm16ToFloat32(pcm: ArrayBuffer): Float32Array {
  const view = new DataView(pcm);
  const out = new Float32Array(pcm.byteLength / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}
