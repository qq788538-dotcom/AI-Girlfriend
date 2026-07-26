class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSamples = Math.max(1, Math.round(sampleRate * 0.04));
    this.chunk = new Float32Array(this.targetSamples);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let inputOffset = 0;
    while (inputOffset < input.length) {
      const copyLength = Math.min(
        input.length - inputOffset,
        this.targetSamples - this.offset,
      );
      this.chunk.set(
        input.subarray(inputOffset, inputOffset + copyLength),
        this.offset,
      );
      this.offset += copyLength;
      inputOffset += copyLength;
      if (this.offset === this.targetSamples) {
        const completed = this.chunk;
        this.chunk = new Float32Array(this.targetSamples);
        this.offset = 0;
        this.port.postMessage(completed, [completed.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
