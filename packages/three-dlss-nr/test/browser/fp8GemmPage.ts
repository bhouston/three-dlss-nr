// Browser page of the hardware-f16 FP8 GEMM check (see run-fp8-gemm-chrome.mjs).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Runs the reference's composed production FP8
// GEMM (reference/OpenDLSS-NR/ports/browser-webgpu/src/matmul, recorded through its own graph.js `Graph.gemm`, exactly
// as test/reference/refKernels.ts does in Node) in a real browser on real f16 hardware, for the cases of
// refKernels.gpu.test.ts with byte-identical inputs, and returns the published bytes. Bundled by the runner with
// esbuild; not part of any vitest project.

import { Graph } from '@ref/graph.js';
import { readBack, requestDevice, storageFrom } from '@ref/gpu.js';
import { Matmul } from '@ref/matmul/index.js';
import { Kernels, Recorder, Tensors } from '@ref/passes.js';

import { fp8BrowserCases, type Fp8Case } from './fp8Cases.js';

const SENTINEL_BYTE = 0xcd;

export interface PageCaseResult {
  label: string;
  /** E4 codes `[rows][outputChannels]` (outputs e4 / dual). */
  e4?: number[];
  /** Half bit patterns (outputs half / dual). */
  half?: number[];
}

export interface PageResult {
  adapter: string;
  features: string[];
  cases: PageCaseResult[];
  /** The reference's GPU-built packed SiLU table (matmul/packed-activation.js), 65536 E4 codes. */
  siluCodes: number[];
}

const alignUp = (value: number, to: number): number => Math.ceil(value / to) * to;
const bytesOf = (data: ArrayBufferView): Uint8Array => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

async function runCase(device: GPUDevice, kernels: any, matmul: any, c: Fp8Case): Promise<PageCaseResult> {
  const tensors = new Tensors(device);
  const recorder = new Recorder(device, kernels, tensors);
  const graph = Object.assign(Object.create(Graph.prototype), {
    device,
    kernels,
    matmul,
    window: null,
    tensors,
    recorder,
    model: null,
    options: {},
    boundaries: new Map(),
  });
  const owned: GPUBuffer[] = [];
  const tensor = (label: string, rows: number, channels: number, format: string, data?: ArrayBufferView) => {
    const t = tensors.allocate(label, rows, channels, format);
    if (data) {
      const padded = new Uint8Array(alignUp(data.byteLength, 4));
      padded.set(bytesOf(data));
      device.queue.writeBuffer(t.buffer, 0, padded);
    }
    return t;
  };
  const fill = (t: any) => device.queue.writeBuffer(t.buffer, 0, new Uint8Array(t.byteLength).fill(SENTINEL_BYTE));

  const input = tensor(`${c.label} in`, c.rows, c.inputChannels, 'e4', c.input);
  const residual = c.residualFormat
    ? tensor(`${c.label} skip`, c.rows, c.outputChannels, c.residualFormat, c.residualData)
    : null;
  const output = c.output !== 'half' ? tensor(`${c.label} e4`, c.rows, c.outputChannels, 'e4') : undefined;
  const outputF16 = c.output !== 'e4' ? tensor(`${c.label} f16`, c.rows, c.outputChannels, 'f16') : undefined;
  for (const t of [output, outputF16]) if (t) fill(t);

  // The weight "stage": matrix bytes, then the skip scales at the next 16-byte boundary (refKernels.ts fp8Matrix).
  const aux = alignUp(c.weights.byteLength, 16);
  const stage = new Uint8Array(alignUp(aux + (c.scale ? c.scale.byteLength : 0) + 4, 4));
  stage.set(c.weights);
  if (c.scale) stage.set(bytesOf(c.scale), aux);
  const weightBuffer = storageFrom(device, stage, 'fp8 matrix');
  owned.push(weightBuffer);

  graph.gemm({
    input,
    weights: { buffer: weightBuffer, byteOffset: 0, k: c.k * c.batches, matrixChannels: c.n, batchK: c.k },
    output,
    outputF16,
    rows: c.rows,
    k: c.k,
    n: c.n,
    batches: c.batches,
    broadcast: c.broadcast,
    partition: c.partition,
    silu: c.silu,
    residual,
    aux: residual ? aux : 0,
    label: c.label,
  });
  await recorder.finish();
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  recorder.encode(encoder);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(`${c.label}: ${error.message}`);
  const result: PageCaseResult = { label: c.label };
  if (output) result.e4 = Array.from(new Uint8Array(await readBack(device, output.buffer, output.validBytes)));
  if (outputF16) {
    result.half = Array.from(new Uint16Array(await readBack(device, outputF16.buffer, outputF16.validBytes)));
  }
  recorder.paramsBuffer?.destroy();
  tensors.destroy();
  for (const buffer of owned) buffer.destroy();
  return result;
}

async function run(): Promise<PageResult> {
  const { device, info: adapter }: { device: GPUDevice; info: any } = await requestDevice();
  if (!device.features.has('shader-f16')) throw new Error('this browser/GPU exposes no shader-f16');
  const kernels = await Kernels.create(device);
  const matmul = await Matmul.create(device);
  const cases: PageCaseResult[] = [];
  for (const c of fp8BrowserCases()) cases.push(await runCase(device, kernels, matmul, c));
  const siluCodes = Array.from(new Uint8Array(await readBack(device, matmul.packedSiluTable, 65536)));
  return {
    siluCodes,
    adapter: `${adapter.vendor ?? '?'} ${adapter.architecture ?? ''} ${adapter.description ?? ''}`.trim(),
    features: [...device.features].toSorted(),
    cases,
  };
}

(globalThis as any).runFp8Gemm = run;
