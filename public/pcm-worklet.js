class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    this.buffer.push(...input);
    const targetSamples = Math.max(1, Math.round(sampleRate * 0.04));
    while (this.buffer.length >= targetSamples) {
      const chunk = new Float32Array(this.buffer.splice(0, targetSamples));
      this.port.postMessage(chunk, [chunk.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
