/**
 * Ambient type declarations for the AudioWorklet global scope.
 *
 * why this file is needed: mic-worklet.ts runs in a completely separate
 * JS realm (AudioWorkletGlobalScope) that TypeScript's default "DOM" lib
 * doesn't model - AudioWorkletProcessor/registerProcessor don't exist in
 * lib.dom.d.ts because they're not available on the main thread. This
 * file teaches the compiler about that separate realm's globals so
 * mic-worklet.ts type-checks, without pulling in the full (and much
 * stricter) "webworker" lib for the entire project.
 */

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: AudioWorkletNodeOptions);
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor,
): void;
