/**
 * Main-thread side of mic capture: sets up getUserMedia + the
 * AudioWorklet, and converts each incoming Float32 frame to 16-bit PCM
 * bytes before handing it to the caller (the WebSocket client).
 *
 * why convert to Int16 here rather than sending Float32 over the wire:
 * the server's app/main.py `_bytes_to_frame` expects 16-bit signed PCM
 * (the universal, half-the-size format every STT/VAD library expects) -
 * matching that on the wire avoids a conversion + format-negotiation
 * step. See docs/03-audio-fundamentals.md for why 16-bit PCM specifically.
 */

export interface MicCapture {
  stop: () => void;
}

export async function startMicCapture(
  onFrame: (pcm16: ArrayBuffer) => void,
  sampleRate = 16000,
): Promise<MicCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const audioContext = new AudioContext({ sampleRate });
  await audioContext.audioWorklet.addModule(new URL("./mic-worklet.ts", import.meta.url));

  const source = audioContext.createMediaStreamSource(stream);
  // Silero VAD accepts 512 samples at 16 kHz (32 ms). Sending its required
  // chunk size prevents the backend from closing the WebSocket on the first
  // microphone frame.
  const frameSize = 512;
  const workletNode = new AudioWorkletNode(audioContext, "mic-capture-processor", {
    processorOptions: { frameSize },
  });

  workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
    onFrame(float32ToPcm16(event.data));
  };

  source.connect(workletNode);
  // why NOT connect workletNode to audioContext.destination: doing so
  // would loop the mic input back out to the speakers (feedback/echo) -
  // we only want to *observe* the audio via the worklet, not play it.

  return {
    stop: () => {
      workletNode.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      audioContext.close();
    },
  };
}

function float32ToPcm16(float32: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32[i]));
    const sample = clamped < 0 ? clamped * 32768 : clamped * 32767;
    view.setInt16(i * 2, sample, true); // little-endian, matches struct.unpack("<h", ...) server-side
  }
  return buffer;
}
