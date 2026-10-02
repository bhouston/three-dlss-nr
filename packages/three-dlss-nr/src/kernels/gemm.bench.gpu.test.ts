// GEMM timings at the 512x512 network's shapes (field 576x512). Opt-in: NR_BENCH=1 pnpm test:gpu.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Wall-clock per dispatch: each kernel is
// dispatched `repeat` times in one compute pass and the queue is drained; the first (compile) run is not timed.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { attributeFromBytes, createHalfVector, createTensor, writeBuffer } from '../tensors.js';
import type { GemmSpec, NRKernel } from '../types.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { createGemmF16 } from './gemmF16.js';
import { createGemmFp8 } from './gemmFp8.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

const random = (length: number, mask: number, seed: number): Uint8Array => {
  let x = seed;
  return Uint8Array.from({ length }, () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 8) & mask;
  });
};

function fp8(rows: number, k: number, n: number, extra: Partial<GemmSpec> = {}): NRKernel {
  const spec: GemmSpec = {
    rows,
    k,
    n,
    batches: 1,
    broadcast: false,
    partition: 0,
    silu: false,
    output: 'e4',
    residual: null,
    label: 'bench',
    ...extra,
  };
  const inputChannels = spec.broadcast ? k : k * spec.batches;
  const outputChannels = n * spec.batches;
  const input = createTensor('in', rows, inputChannels, 'e4');
  writeBuffer(input, random(rows * inputChannels, 0x77, 7));
  const weights = {
    attribute: attributeFromBytes(random(k * n * spec.batches, 0x47, 9), 4),
    k: k * spec.batches,
    n,
    batchK: k,
  };
  const output = spec.output !== 'half' ? createTensor('out', rows, outputChannels, 'e4') : undefined;
  const outputF16 = spec.output !== 'e4' ? createTensor('out16', rows, outputChannels, 'f16') : undefined;
  const residual = spec.residual ? createTensor('skip', rows, outputChannels, spec.residual.format) : undefined;
  const scale = spec.residual ? createHalfVector(new Uint16Array(n).fill(0x3800)) : undefined;
  return createGemmFp8(spec, { input, weights, output, outputF16, residual, scale });
}

/**
 * GPU time of one dispatch: the best of `trials` compute passes of `repeat` dispatches each, from timestamp queries
 * (three's `trackTimestamp`) when the device has them, else wall clock. Best-of, because other processes may share
 * the GPU.
 */
async function time(kernel: NRKernel, repeat = 4, trials = 25): Promise<number> {
  const renderer = gpu.renderer;
  const timestamps = gpu.device.features.has('timestamp-query');
  renderer.backend.trackTimestamp = timestamps;
  renderer.compute([kernel.node]);
  await gpu.device.queue.onSubmittedWorkDone();
  if (timestamps) await renderer.resolveTimestampsAsync('compute');
  let best = Infinity;
  for (let trial = 0; trial < trials; ++trial) {
    const start = performance.now();
    renderer.compute(Array.from({ length: repeat }, () => kernel.node));
    await gpu.device.queue.onSubmittedWorkDone();
    const ms = timestamps ? await renderer.resolveTimestampsAsync('compute') : performance.now() - start;
    best = Math.min(best, ms / repeat);
  }
  renderer.backend.trackTimestamp = false;
  return best;
}

describe.skipIf(!process.env.NR_BENCH)('GEMM timings at 512x512 (field 576x512)', () => {
  it('FP8 and f16 GEMMs per dispatch', async () => {
    const L0 = 576 * 512;
    const cases: [string, () => NRKernel, number][] = [
      ['L0 expand 32->128 SiLU', () => fp8(L0, 32, 128, { silu: true }), 7],
      ['L0 contract 128->32 dual f16 skip', () => fp8(L0, 128, 32, { output: 'dual', residual: { format: 'f16' } }), 6],
      ['L0 qkv 32->96 half', () => fp8(L0, 32, 96, { output: 'half' }), 6],
      ['L0 projection 32 dual f16 skip', () => fp8(L0, 32, 32, { output: 'dual', residual: { format: 'f16' } }), 6],
      ['L1 expert expand 64 (2x128)', () => fp8(L0 / 4, 64, 128, { batches: 2, broadcast: true, silu: true }), 8],
      ['L1 expert contract (2x 128->32)', () => fp8(L0 / 4, 128, 32, { batches: 2 }), 8],
      ['L1 qkv 64->192', () => fp8(L0 / 4, 64, 192, { output: 'half' }), 8],
      ['L3 expert expand 256 (8x128)', () => fp8(L0 / 64, 256, 128, { batches: 8, broadcast: true, silu: true }), 16],
      ['L4 split layer0 512', () => fp8(L0 / 256, 512, 512), 16],
      ['ViT contract 4096->1024 p1024', () => fp8(96, 4096, 1024, { partition: 1024, residual: { format: 'e4' } }), 8],
      [
        'adapter f16 16->32',
        () => {
          const input = createTensor('features', L0, 16, 'f16');
          const weights = {
            attribute: attributeFromBytes(new Uint16Array(16 * 32).fill(0x3000)),
            k: 16,
            n: 32,
            paddedN: 32,
          };
          return createGemmF16(
            { rows: L0, k: 16, n: 32, label: 'adapter' },
            { input, weights, output: createTensor('e4', L0, 32, 'e4'), outputF16: createTensor('f16', L0, 32, 'f16') },
          );
        },
        1,
      ],
      [
        'head f16 32->4',
        () => {
          const input = createTensor('post', L0, 32, 'f16');
          const weights = {
            attribute: attributeFromBytes(new Uint16Array(32 * 16).fill(0x3000)),
            k: 32,
            n: 4,
            paddedN: 16,
          };
          return createGemmF16(
            { rows: L0, k: 32, n: 4, label: 'head' },
            { input, weights, outputF32: createTensor('head', L0, 4, 'f32') },
          );
        },
        1,
      ],
    ];
    const lines: string[] = [];
    for (const [name, make, perFrame] of cases) {
      const ms = await time(make());
      lines.push(`${name.padEnd(40)} ${ms.toFixed(3).padStart(8)} ms  x${perFrame}/frame`);
    }
    const info = gpu.adapter.info;
    console.info(
      `[bench] ${info.vendor} ${info.description}; timestamps: ${gpu.device.features.has('timestamp-query')}\n${lines.join('\n')}`,
    );
    expect(lines.length).toBe(cases.length);
  });
});
