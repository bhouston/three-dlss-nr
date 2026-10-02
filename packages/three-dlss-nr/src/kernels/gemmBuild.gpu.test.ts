// Build budget of the GEMMs: every distinct GEMM shape of the 64x64 network (field 336x320) builds and compiles
// (D3D12/FXC locally) with one program per shape. The compile time is printed, and the time bound is only a hang guard.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). The shapes follow graph.js `Graph.gemm`
// calls (blocks, transitions, ViT) at the levels of a 336x320 field; chunk E's graph test covers the exact list.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { attributeFromBytes, createHalfVector, createTensor } from '../tensors.js';
import { kernelWGSL } from '../tsl/KernelBuilder.js';
import type { GemmSpec, NRKernel } from '../types.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { normalizeWGSL } from '../../test/wgsl.js';
import { createGemmF16 } from './gemmF16.js';
import { createGemmFp8 } from './gemmFp8.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

type Shape = Omit<GemmSpec, 'label' | 'batches' | 'broadcast' | 'partition' | 'silu' | 'residual'> &
  Partial<Pick<GemmSpec, 'batches' | 'broadcast' | 'partition' | 'silu' | 'residual'>>;

/** The GEMM shapes of one network whose level-0 field has `rows0` rows (levels quarter the rows) and `tokens` ViT tokens. */
export function networkGemmShapes(rows0: number, tokens: number): Shape[] {
  const level = (l: number) => rows0 / 4 ** l;
  const f16 = { format: 'f16' as const };
  const e4 = { format: 'e4' as const };
  const shapes: Shape[] = [];
  // 32-channel blocks (level 0).
  const r0 = level(0);
  shapes.push(
    { rows: r0, k: 32, n: 128, silu: true, output: 'e4' },
    { rows: r0, k: 128, n: 32, output: 'dual', residual: f16 },
    { rows: r0, k: 32, n: 96, output: 'half' },
    { rows: r0, k: 32, n: 32, output: 'dual', residual: f16 },
    { rows: r0, k: 32, n: 32, output: 'e4', residual: f16 },
  );
  // Expert blocks, levels 1-3.
  for (const [l, c] of [
    [1, 64],
    [2, 128],
    [3, 256],
  ]) {
    const rows = level(l);
    const experts = c / 32;
    shapes.push(
      { rows, k: c, n: 128, batches: experts, broadcast: true, silu: true, output: 'e4' },
      { rows, k: 128, n: 32, batches: experts, output: 'e4' },
      { rows, k: c, n: c, output: 'dual', residual: e4 },
      { rows, k: c, n: 3 * c, output: 'half' },
      { rows, k: c, n: c, output: 'e4', residual: e4 },
      { rows, k: c, n: c, output: 'dual', residual: e4 },
    );
  }
  // Split blocks, level 4.
  const r4 = level(4);
  shapes.push(
    { rows: r4, k: 512, n: 512, output: 'e4' },
    { rows: r4, k: 64, n: 256, batches: 8, silu: true, output: 'e4' },
    { rows: r4, k: 256, n: 64, batches: 8, output: 'e4' },
    { rows: r4, k: 512, n: 512, output: 'e4', residual: e4 },
    { rows: r4, k: 512, n: 1536, output: 'half' },
  );
  // Transitions down and up.
  for (const [l, c] of [
    [1, 32],
    [2, 64],
    [3, 128],
    [4, 256],
  ]) {
    shapes.push({ rows: level(l), k: c, n: 2 * c, output: 'e4' });
    shapes.push({ rows: level(l), k: 2 * c, n: c, output: 'half' });
  }
  // ViT and its transitions.
  shapes.push(
    { rows: tokens, k: 512, n: 1024, output: 'e4' },
    { rows: tokens, k: 1024, n: 4096, silu: true, output: 'e4' },
    { rows: tokens, k: 4096, n: 1024, partition: 1024, output: 'e4', residual: e4 },
    { rows: tokens, k: 1024, n: 3072, partition: 512, output: 'half' },
    { rows: tokens, k: 1024, n: 1024, partition: 256, output: 'e4', residual: e4 },
    { rows: tokens, k: 1024, n: 512, partition: 256, output: 'half' },
  );
  return shapes;
}

function build(shape: Shape, label: string): NRKernel {
  const spec: GemmSpec = {
    batches: 1,
    broadcast: false,
    partition: 0,
    silu: false,
    residual: null,
    label,
    ...shape,
  };
  const outputChannels = spec.n * spec.batches;
  const inputChannels = spec.broadcast ? spec.k : spec.k * spec.batches;
  return createGemmFp8(spec, {
    input: createTensor(`${label} in`, spec.rows, inputChannels, 'e4'),
    weights: {
      attribute: attributeFromBytes(new Uint8Array(spec.k * spec.n * spec.batches), 4),
      k: spec.k * spec.batches,
      n: spec.n,
      batchK: spec.k,
    },
    output: spec.output !== 'half' ? createTensor(`${label} out`, spec.rows, outputChannels, 'e4') : undefined,
    outputF16: spec.output !== 'e4' ? createTensor(`${label} raw`, spec.rows, outputChannels, 'f16') : undefined,
    residual: spec.residual
      ? createTensor(`${label} skip`, spec.rows, outputChannels, spec.residual.format)
      : undefined,
    scale: spec.residual ? createHalfVector(new Uint16Array(spec.n)) : undefined,
  });
}

describe('GEMM build budget (64x64 network, field 336x320)', () => {
  it('compiles every distinct GEMM shape, one program each', { timeout: 300_000 }, async () => {
    const shapes = networkGemmShapes(336 * 320, 64);
    const kernels = shapes.map((shape, i) => build(shape, `gemm ${i}`));
    kernels.push(
      createGemmF16(
        { rows: 336 * 320, k: 16, n: 32, label: 'adapter' },
        {
          input: createTensor('features', 336 * 320, 16, 'f16'),
          weights: { attribute: attributeFromBytes(new Uint16Array(16 * 32)), k: 16, n: 32, paddedN: 32 },
          output: createTensor('adapter e4', 336 * 320, 32, 'e4'),
          outputF16: createTensor('adapter f16', 336 * 320, 32, 'f16'),
        },
      ),
      createGemmF16(
        { rows: 336 * 320, k: 32, n: 4, label: 'head' },
        {
          input: createTensor('post', 336 * 320, 32, 'f16'),
          weights: { attribute: attributeFromBytes(new Uint16Array(32 * 16)), k: 32, n: 4, paddedN: 16 },
          outputF32: createTensor('head', 336 * 320, 4, 'f32'),
        },
      ),
    );
    // A second dispatch of an already-present shape (another block's weights).
    kernels.push(build(shapes[0], 'gemm again'));

    const start = performance.now(); // unreliable when the GPU / CPU is shared
    await gpu.renderer.compileComputeAsync(kernels.map((k) => k.node));
    const seconds = (performance.now() - start) / 1000;
    const programs = new Set(kernels.map((k) => normalizeWGSL(kernelWGSL(gpu.renderer, k))));
    const lines = Math.max(...kernels.map((k) => kernelWGSL(gpu.renderer, k).split('\n').length));
    console.info(
      `[gemm build] ${kernels.length} kernels, ${programs.size} programs, ${seconds.toFixed(2)} s, ` +
        `largest ${lines} WGSL lines`,
    );
    // One program per distinct spec (the shape list repeats a few, e.g. equal transition shapes on two levels), plus
    // at most a couple: three caches each layout `Fn` per backend (NodeBuilder.buildFunctionNode), and the first build
    // of a helper records its nested helpers as includes of the caller, so the very first kernels of a process emit
    // the helper functions in a different order than every later build of the same shape (a three 0.186 quirk).
    const specs = new Set(shapes.map((shape) => JSON.stringify(shape)));
    expect(programs.size).toBeGreaterThanOrEqual(specs.size + 2);
    expect(programs.size).toBeLessThanOrEqual(specs.size + 2 + 2);
    // In steady state a repeated shape shares its program.
    const later = [build(shapes[0], 'gemm later a'), build(shapes[0], 'gemm later b')];
    await gpu.renderer.compileComputeAsync(later.map((k) => k.node));
    expect(normalizeWGSL(kernelWGSL(gpu.renderer, later[1]))).toBe(normalizeWGSL(kernelWGSL(gpu.renderer, later[0])));
    expect(normalizeWGSL(kernelWGSL(gpu.renderer, later[0]))).toBe(
      normalizeWGSL(kernelWGSL(gpu.renderer, kernels[kernels.length - 1])),
    );
    // Generous: FXC (D3D12 without DXC) compiles these slowly, and the dev GPU may be shared. Not a performance gate.
    expect(seconds).toBeLessThan(240);
  });
});
