import * as ort from "./vendor/onnxruntime-web-1.22.0/ort.wasm.min.mjs";

const SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 512;
const CONTEXT_SAMPLES = 64;
const STATE_SAMPLES = 2 * 1 * 128;
const MODEL_URL = "/vendor/silero-vad-v6.2/silero_vad.onnx";
const WASM_ROOT = "/vendor/onnxruntime-web-1.22.0/";

ort.env.wasm.wasmPaths = WASM_ROOT;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

function appendFloat32(left, right) {
  if (left.length === 0) return right.slice();
  const joined = new Float32Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

function resampleTo16k(samples, inputSampleRate) {
  if (inputSampleRate === SAMPLE_RATE) return samples.slice();
  if (!Number.isFinite(inputSampleRate) || inputSampleRate <= 0) {
    throw new Error(`Invalid input sample rate: ${inputSampleRate}`);
  }

  const outputLength = Math.max(
    1,
    Math.round((samples.length * SAMPLE_RATE) / inputSampleRate),
  );
  const output = new Float32Array(outputLength);
  const sourceStep = inputSampleRate / SAMPLE_RATE;
  for (let index = 0; index < outputLength; index += 1) {
    const sourcePosition = index * sourceStep;
    const leftIndex = Math.min(Math.floor(sourcePosition), samples.length - 1);
    const rightIndex = Math.min(leftIndex + 1, samples.length - 1);
    const fraction = sourcePosition - leftIndex;
    output[index] =
      samples[leftIndex] * (1 - fraction) + samples[rightIndex] * fraction;
  }
  return output;
}

export class LocalSileroVad {
  static async create() {
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new LocalSileroVad(session);
  }

  constructor(session) {
    this.session = session;
    this.reset();
  }

  reset() {
    this.state = new Float32Array(STATE_SAMPLES);
    this.context = new Float32Array(CONTEXT_SAMPLES);
    this.pending = new Float32Array();
  }

  async process(samples, inputSampleRate) {
    this.pending = appendFloat32(
      this.pending,
      resampleTo16k(samples, inputSampleRate),
    );
    const probabilities = [];

    while (this.pending.length >= FRAME_SAMPLES) {
      const frame = this.pending.slice(0, FRAME_SAMPLES);
      this.pending = this.pending.slice(FRAME_SAMPLES);

      const modelInput = new Float32Array(CONTEXT_SAMPLES + FRAME_SAMPLES);
      modelInput.set(this.context);
      modelInput.set(frame, CONTEXT_SAMPLES);

      const result = await this.session.run({
        input: new ort.Tensor("float32", modelInput, [1, modelInput.length]),
        state: new ort.Tensor("float32", this.state, [2, 1, 128]),
        sr: new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), [1]),
      });

      this.state = Float32Array.from(result.stateN.data);
      this.context = modelInput.slice(modelInput.length - CONTEXT_SAMPLES);
      probabilities.push(Number(result.output.data[0]));
    }

    return probabilities;
  }
}

export const SILERO_VAD_VERSION = "6.2";
