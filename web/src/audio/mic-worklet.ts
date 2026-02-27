/**
 * AudioWorkletProcessor that runs on the browser's dedicated audio
 * rendering thread, not the main JS thread.
 *
 * why an AudioWorklet instead of the older ScriptProcessorNode: audio
 * worklets run on a real-time-priority audio thread separate from the
 * main thread, so mic capture keeps working smoothly even if React is
 * busy re-rendering. ScriptProcessorNode runs on the main thread and is
 * both deprecated and prone to audible glitches under load - see
 * docs/12-frontend-audio.md.
 *
 * This file is loaded via `audioContext.audioWorklet.addModule(...)` -
 * it does NOT run in the same JS context as the rest of the app (no
 * access to window, React state, etc). It only talks to the main thread
 * via postMessage/port, which is why capture.ts (the main-thread side)
 * exists as a separate file.
 */

class MicCaptureProcessor extends AudioWorkletProcessor {
  // why buffer up to a fixed frame size instead of forwarding every
  // 128-sample AudioWorklet render quantum immediately: 128 samples @
  // 16kHz is only 8ms. We accumulate to 512-sample frames because that is
  // the chunk size Silero VAD requires at the configured 16 kHz sample rate.
  private buffer: Float32Array;
  private writeIndex = 0;
  private readonly frameSize: number;

  constructor(options?: AudioWorkletNodeOptions) {
    super();
    this.frameSize = (options?.processorOptions?.frameSize as number) ?? 512; // 32ms @ 16kHz
    this.buffer = new Float32Array(this.frameSize);
  }

  process(inputs: Float32Array[][]): boolean {
    const channelData = inputs[0]?.[0];
    if (!channelData) return true;

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.writeIndex++] = channelData[i];
      if (this.writeIndex >= this.frameSize) {
        // transfer ownership of the underlying buffer to avoid a copy -
        // we allocate a fresh buffer right after for the next frame.
        this.port.postMessage(this.buffer);
        this.buffer = new Float32Array(this.frameSize);
        this.writeIndex = 0;
      }
    }
    return true; // keep the processor alive
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
