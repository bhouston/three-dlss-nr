// The TSL FP8 GEMM, byte for byte: against the CPU oracle on every graph role (everywhere), and against the reference
// port's composed kernel (wherever the reference runs with trustworthy f16: see test/reference/refKernels.ts).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Each role is a call graph.js `Graph.gemm`
// makes, at a small row count but with the role's real K, flags and strides (N is reduced for the widest ViT roles to
// keep the CPU oracle fast; N only changes the fragment addressing, which other roles cover at full width).
// Outputs are prefilled with a sentinel and compared whole, padding rows included (R6). The inputs are hostile on
// purpose: ~6% zeros of both signs, E4 subnormals, NaN codes in activations and weights (both decode to 0), and skip
// values small enough that `residual * scale` rounds to a signed zero.

import { createHash } from 'node:crypto';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createHalfVector, createTensor, readBuffer, writeBuffer, attributeFromBytes } from '../tensors.js';
import { kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import type { FP8Matrix, NRTensor } from '../types.js';
import { describeMismatches, diffArrays, fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { GEMM_ROLES, gemmData, oracleGemmFp8, type GemmData, type GemmRole } from '../../test/oracle/gemm.js';
import { RefKernels } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons, expectNoApproximations, normalizeWGSL } from '../../test/wgsl.js';
import { createGemmFp8, gemmFp8Dispatch } from './gemmFp8.js';

let gpu: GpuTestContext & { dispose(): void };
let ref: RefKernels;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  ref = await RefKernels.create(gpu.device);
});

afterAll(() => {
  ref?.destroy();
  gpu?.dispose();
});

const tensorWith = (label: string, rows: number, channels: number, format: 'e4' | 'f16', data?: ArrayBufferView) => {
  const tensor = createTensor(label, rows, channels, format);
  if (data) writeBuffer(tensor, data);
  return tensor;
};

export interface OurGemm {
  output?: NRTensor;
  outputF16?: NRTensor;
  kernel: ReturnType<typeof createGemmFp8>;
}

/** Our kernel over `data`, outputs prefilled with the sentinel. */
export function ourGemm(data: GemmData, label = data.spec.label): OurGemm {
  const { spec } = data;
  const input = tensorWith(`${label} in`, spec.rows, data.inputChannels, 'e4', data.input);
  const weights: FP8Matrix = {
    attribute: attributeFromBytes(data.weights, 4),
    k: spec.k * spec.batches,
    n: spec.n,
    batchK: spec.k,
  };
  const output = spec.output !== 'half' ? tensorWith(`${label} e4`, spec.rows, data.outputChannels, 'e4') : undefined;
  const outputF16 =
    spec.output !== 'e4' ? tensorWith(`${label} f16`, spec.rows, data.outputChannels, 'f16') : undefined;
  for (const tensor of [output, outputF16]) if (tensor) fillSentinel(tensor);
  const residual = spec.residual
    ? tensorWith(`${label} skip`, spec.rows, data.outputChannels, spec.residual.format, data.residual)
    : undefined;
  const scale = data.scale ? createHalfVector(data.scale) : undefined;
  return { output, outputF16, kernel: createGemmFp8(spec, { input, weights, output, outputF16, residual, scale }) };
}

/** A whole allocation: `bytes` from byte 0, the sentinel after them. */
function fill(tensor: NRTensor, bytes: Uint8Array): Uint8Array {
  const whole = new Uint8Array(tensor.byteLength).fill(SENTINEL_BYTE);
  whole.set(bytes);
  return whole;
}

/** Expected whole allocations: oracle bytes over the logical rows, the sentinel past them. */
function expectedAllocations(data: GemmData, ours: OurGemm) {
  const { spec } = data;
  const result = oracleGemmFp8({
    ...spec,
    output: spec.output,
    residual: spec.residual ? { format: spec.residual.format, data: data.residual! } : undefined,
    input: data.input,
    inputChannels: data.inputChannels,
    weights: data.weights,
    scale: data.scale,
    outputChannels: data.outputChannels,
  });
  return {
    e4: ours.output ? fill(ours.output, result.e4!) : undefined,
    f16: ours.outputF16 ? fill(ours.outputF16, new Uint8Array(result.half!.buffer)) : undefined,
  };
}

const halves = (bytes: Uint8Array) => new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);

describe('FP8 GEMM (TSL) vs oracleGemmFp8, every graph role', () => {
  it.for(GEMM_ROLES)('$label', async (role) => {
    const data = gemmData(role);
    const ours = ourGemm(data);
    await runKernels(gpu.renderer, [ours.kernel]);
    const expected = expectedAllocations(data, ours);
    if (ours.output) {
      const r = diffArrays(await readBuffer(gpu.renderer, ours.output, { validOnly: false }), expected.e4!);
      expect(r.mismatches, describeMismatches(`${role.label} e4`, r, 2)).toBe(0);
    }
    if (ours.outputF16) {
      const got = halves(await readBuffer(gpu.renderer, ours.outputF16, { validOnly: false }));
      const r = diffArrays(got, halves(expected.f16!));
      expect(r.mismatches, describeMismatches(`${role.label} f16`, r, 4)).toBe(0);
    }
  });

  it('publishes -0 as 0x00 and a zero result as +0 (all-zero products, signed-zero skips)', async () => {
    const role: GemmRole = { label: 'zeros', rows: 32, k: 32, n: 32, output: 'dual', residual: 'f16' };
    const data = gemmData(role);
    // Half the rows multiply zeros of both signs; their skips are -0 or tiny negatives that round to -0.
    for (let row = 0; row < 16; ++row) {
      for (let k = 0; k < 32; ++k) data.input[row * 32 + k] = k % 2 ? 0x80 : 0;
      for (let c = 0; c < 32; ++c) (data.residual as Uint16Array)[row * 32 + c] = c % 2 ? 0x8000 : 0x8001;
    }
    data.scale!.fill(0x2c00); // 2^-4: 2^-24 * 2^-4 rounds to -0
    const ours = ourGemm(data);
    await runKernels(gpu.renderer, [ours.kernel]);
    const e4 = await readBuffer(gpu.renderer, ours.output!);
    const f16 = halves(await readBuffer(gpu.renderer, ours.outputF16!));
    for (let i = 0; i < 16 * 32; ++i) {
      expect(e4[i], `e4 [${i}]`).toBe(0);
      expect(f16[i], `f16 [${i}]`).toBe(0);
    }
    const expected = expectedAllocations(data, ours);
    expect(diffArrays(e4, expected.e4!.subarray(0, e4.length)).mismatches).toBe(0);
  });

  it.skipIf(process.env.CI)('folds more than 65535 row tiles over y and z (local)', async () => {
    // 65536 * 32 + 40 rows: z = 2, the second z slab holds 2 real tiles and 65533 empty ones. Rows repeat every 64,
    // so the expected output is the oracle's 64 rows tiled.
    const period = 64;
    const rows = 65536 * 32 + 40;
    const role: GemmRole = { label: 'folded', rows: period, k: 64, n: 32, output: 'dual', residual: 'e4' };
    const small = gemmData(role);
    const tile = <T extends Uint8Array | Uint16Array>(source: T, width: number): T => {
      const out = new (source.constructor as any)(rows * width) as T;
      for (let row = 0; row < rows; row += period)
        out.set(source.subarray(0, Math.min(period, rows - row) * width), row * width);
      return out;
    };
    const data: GemmData = {
      ...small,
      spec: { ...small.spec, rows },
      input: tile(small.input, 64),
      residual: tile(small.residual as Uint8Array, 32),
    };
    expect(gemmFp8Dispatch(data.spec)).toEqual([1, 65535, 2]);
    const ours = ourGemm(data);
    await runKernels(gpu.renderer, [ours.kernel]);
    const expected = oracleGemmFp8({
      ...small.spec,
      residual: { format: 'e4', data: small.residual! },
      input: small.input,
      weights: small.weights,
      scale: small.scale,
    });
    const e4 = await readBuffer(gpu.renderer, ours.output!, { validOnly: false });
    const f16 = halves(await readBuffer(gpu.renderer, ours.outputF16!, { validOnly: false }));
    let mismatches = 0;
    for (let i = 0; i < rows * 32; ++i) {
      const j = i % (period * 32);
      if (e4[i] !== expected.e4![j] || f16[i] !== expected.half![j]) mismatches += 1;
    }
    for (let i = rows * 32; i < e4.length; ++i) if (e4[i] !== SENTINEL_BYTE) mismatches += 1;
    expect(mismatches).toBe(0);
  });
});

describe('FP8 GEMM (TSL) vs the reference composed kernel', () => {
  it.for(GEMM_ROLES)('$label', async (role, context) => {
    const reason = ref.unavailableReason('gemm_fp8');
    if (reason) return context.skip(reason);
    const data = gemmData(role);
    const { spec } = data;
    const ours = ourGemm(data);
    await runKernels(gpu.renderer, [ours.kernel]);

    const input = ref.tensor(`${role.label} in`, spec.rows, data.inputChannels, 'e4', data.input);
    const output = ours.output ? ref.tensor(`${role.label} e4`, spec.rows, data.outputChannels, 'e4') : undefined;
    const outputF16 = ours.outputF16
      ? ref.tensor(`${role.label} f16`, spec.rows, data.outputChannels, 'f16')
      : undefined;
    for (const tensor of [output, outputF16]) if (tensor) ref.fill(tensor, SENTINEL_BYTE);
    const residual = spec.residual
      ? ref.tensor(`${role.label} skip`, spec.rows, data.outputChannels, spec.residual.format, data.residual)
      : null;
    ref.gemm({
      input,
      weights: ref.fp8Matrix({
        bytes: data.weights,
        k: spec.k * spec.batches,
        n: spec.n,
        batchK: spec.k,
        scales: data.scale,
      }),
      output,
      outputF16,
      rows: spec.rows,
      k: spec.k,
      n: spec.n,
      batches: spec.batches,
      broadcast: spec.broadcast,
      partition: spec.partition,
      silu: spec.silu,
      residual,
      label: role.label,
    });
    await ref.run();
    if (output) {
      const r = diffArrays(
        await readBuffer(gpu.renderer, ours.output!, { validOnly: false }),
        await ref.read(output, { validOnly: false }),
      );
      expect(r.mismatches, describeMismatches(`${role.label} e4`, r, 2)).toBe(0);
    }
    if (outputF16) {
      const r = diffArrays(
        halves(await readBuffer(gpu.renderer, ours.outputF16!, { validOnly: false })),
        halves(await ref.read(outputF16, { validOnly: false })),
      );
      expect(r.mismatches, describeMismatches(`${role.label} f16`, r, 4)).toBe(0);
    }
  });
});

describe('FP8 GEMM WGSL', () => {
  it('compares integers as integers, uses no approximations, and shares one program per shape', () => {
    for (const role of GEMM_ROLES.slice(0, 4)) {
      const wgsl = kernelWGSL(gpu.renderer, ourGemm(gemmData(role)).kernel);
      expectIntegerComparisons(wgsl);
      expectNoApproximations(wgsl);
    }
    // Same shape, different buffers and label: identical WGSL (R15), so three compiles it once.
    const role = GEMM_ROLES[1];
    const a = kernelWGSL(gpu.renderer, ourGemm(gemmData(role), 'block 1 contract').kernel);
    const b = kernelWGSL(gpu.renderer, ourGemm(gemmData(role), 'block 2 contract').kernel);
    expect(normalizeWGSL(b)).toBe(normalizeWGSL(a));
  });

  it('matches the snapshot (codegen drift on a three bump shows here)', () => {
    const digest = (role: GemmRole) =>
      createHash('sha256')
        .update(normalizeWGSL(kernelWGSL(gpu.renderer, ourGemm(gemmData(role)).kernel)))
        .digest('hex')
        .slice(0, 16);
    const hashes = Object.fromEntries(GEMM_ROLES.map((role) => [role.label, digest(role)]));
    expect(hashes).toMatchSnapshot();
    const contract = GEMM_ROLES[1];
    expect(normalizeWGSL(kernelWGSL(gpu.renderer, ourGemm(gemmData(contract)).kernel))).toMatchSnapshot();
  });
});
