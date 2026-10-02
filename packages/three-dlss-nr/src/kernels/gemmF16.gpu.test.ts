// The TSL f16 GEMM (input adapter and head), byte for byte: against oracleGemmF16 everywhere, and against the
// reference's shaders/gemm_f16.wgsl wherever it compiles (not under FXC on D3D12: see src/README-internals.md).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Outputs are prefilled with a sentinel and
// compared whole (R6). Inputs span the half range down to subnormals, with signed zeros.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { alignUp } from '../geometry.js';
import * as oracle from '../numerics/oracle.js';
import { attributeFromBytes, createTensor, readBuffer, writeBuffer } from '../tensors.js';
import { kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import type { F16Matrix, NRTensor } from '../types.js';
import { describeMismatches, diffArrays, fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { oracleGemmF16 } from '../../test/oracle/gemm.js';
import { RefKernels } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons, expectNoApproximations, normalizeWGSL } from '../../test/wgsl.js';
import { createGemmF16 } from './gemmF16.js';

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

interface F16Role {
  label: string;
  rows: number;
  k: number;
  n: number;
  outputs: ('e4' | 'f16' | 'f32')[];
}

const ROLES: F16Role[] = [
  { label: 'input adapter 16->32 (E4 + f16)', rows: 50, k: 16, n: 32, outputs: ['e4', 'f16'] },
  { label: 'head 32->4 (f32)', rows: 77, k: 32, n: 4, outputs: ['f32'] },
  { label: 'all three outputs 32->16', rows: 64, k: 32, n: 16, outputs: ['e4', 'f16', 'f32'] },
];

function roleData(role: F16Role, seed: number) {
  const rng = new oracle.Xorshift(seed);
  const half = (maxExponent: number) => {
    const draw = rng.next();
    if (draw % 17 === 0) return draw & 0x8000;
    return (draw & 0x8000) | (((draw >>> 16) % maxExponent) << 10) | (draw & 0x3ff);
  };
  const paddedN = alignUp(role.n, 16);
  return {
    paddedN,
    input: Uint16Array.from({ length: role.rows * role.k }, () => half(17)),
    weights: Uint16Array.from({ length: role.k * paddedN }, () => half(16)),
  };
}

function ours(role: F16Role, data: ReturnType<typeof roleData>, label = role.label) {
  const input = createTensor(`${label} in`, role.rows, role.k, 'f16');
  writeBuffer(input, data.input);
  const weights: F16Matrix = {
    attribute: attributeFromBytes(data.weights),
    k: role.k,
    n: role.n,
    paddedN: data.paddedN,
  };
  const out = (format: 'e4' | 'f16' | 'f32'): NRTensor | undefined => {
    if (!role.outputs.includes(format)) return undefined;
    const tensor = createTensor(`${label} ${format}`, role.rows, role.n, format);
    fillSentinel(tensor);
    return tensor;
  };
  const outputs = { output: out('e4'), outputF16: out('f16'), outputF32: out('f32') };
  const kernel = createGemmF16({ rows: role.rows, k: role.k, n: role.n, label }, { input, weights, ...outputs });
  return { kernel, ...outputs };
}

const whole = (tensor: NRTensor, valid: Uint8Array) => {
  const bytes = new Uint8Array(tensor.byteLength).fill(SENTINEL_BYTE);
  bytes.set(valid);
  return bytes;
};
const bytesOf = (view: ArrayBufferView) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

describe('f16 GEMM (TSL) vs oracleGemmF16', () => {
  it.for(ROLES.map((role, i) => ({ ...role, seed: 0x9e3779b9 + i })))('$label', async (role) => {
    const data = roleData(role, role.seed);
    const k = ours(role, data);
    await runKernels(gpu.renderer, [k.kernel]);
    const expected = oracleGemmF16({ rows: role.rows, k: role.k, n: role.n, ...data });
    const pairs: [NRTensor | undefined, Uint8Array, number][] = [
      [k.output, expected.e4, 1],
      [k.outputF16, bytesOf(expected.half), 2],
      [k.outputF32, bytesOf(expected.f32), 4],
    ];
    for (const [tensor, valid, width] of pairs) {
      if (!tensor) continue;
      const got = await readBuffer(gpu.renderer, tensor, { validOnly: false });
      const want = whole(tensor, valid);
      const view = (b: Uint8Array) =>
        width === 1
          ? b
          : width === 2
            ? new Uint16Array(b.buffer, b.byteOffset, b.length / 2)
            : new Uint32Array(b.buffer, b.byteOffset, b.length / 4);
      const r = diffArrays(view(got), view(want));
      expect(r.mismatches, describeMismatches(`${role.label} ${tensor.format}`, r, width * 2)).toBe(0);
    }
  });

  it('WGSL: integer comparisons, no approximations, one program per shape', () => {
    const role = ROLES[0];
    const a = kernelWGSL(gpu.renderer, ours(role, roleData(role, 1), 'adapter a').kernel);
    const b = kernelWGSL(gpu.renderer, ours(role, roleData(role, 2), 'adapter b').kernel);
    expectIntegerComparisons(a);
    expectNoApproximations(a);
    expect(normalizeWGSL(b)).toBe(normalizeWGSL(a));
  });
});

describe('f16 GEMM (TSL) vs the reference gemm_f16.wgsl', () => {
  it.for(ROLES.map((role, i) => ({ ...role, seed: 0x7f4a7c15 + i })))('$label', async (role, context) => {
    const reason = ref.unavailableReason('gemm_f16');
    if (reason) return context.skip(reason);
    const data = roleData(role, role.seed);
    const k = ours(role, data);
    await runKernels(gpu.renderer, [k.kernel]);
    const input = ref.tensor(`${role.label} in`, role.rows, role.k, 'f16', data.input);
    const refOut = (format: 'e4' | 'f16' | 'f32') => {
      if (!role.outputs.includes(format)) return undefined;
      const tensor = ref.tensor(`${role.label} ${format}`, role.rows, role.n, format);
      ref.fill(tensor, SENTINEL_BYTE);
      return tensor;
    };
    const outputs = { output: refOut('e4'), outputF16: refOut('f16'), outputF32: refOut('f32') };
    ref.gemmF16({
      input,
      weights: ref.buffer(data.weights),
      paddedN: data.paddedN,
      ...outputs,
      rows: role.rows,
      k: role.k,
      n: role.n,
      label: role.label,
    });
    await ref.run();
    for (const [mine, theirs] of [
      [k.output, outputs.output],
      [k.outputF16, outputs.outputF16],
      [k.outputF32, outputs.outputF32],
    ] as const) {
      if (!mine || !theirs) continue;
      const r = diffArrays(
        await readBuffer(gpu.renderer, mine, { validOnly: false }),
        await ref.read(theirs, { validOnly: false }),
      );
      expect(r.mismatches, describeMismatches(`${role.label} ${mine.format}`, r, 2)).toBe(0);
    }
  });
});
