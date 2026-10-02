// The window attention kernel (`attend_window_tiled`) against the CPU oracle of the composed reference WGSL
// (test/oracle/attention.ts) and, where the reference can run (it needs shader-f16), against the reference itself.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { describeMismatches, diffArrays, fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { oracleWindowAttention } from '../../test/oracle/attention.js';
import { makePrior, makeQkv, makeScales, type QkvOptions } from '../../test/oracle/attentionInputs.js';
import { RefKernels } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons, expectNoApproximations } from '../../test/wgsl.js';
import { windowPhase } from '../geometry.js';
import { createF32Vector, createHalfVector, createTensor, readBuffer, writeBuffer } from '../tensors.js';
import { kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import { createWindowAttention, windowAttentionDispatch, windowAttentionWorkgroupBytes } from './windowAttention.js';
import type { WindowQueries } from './windowAttention.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

interface WindowCase {
  width: number;
  height: number;
  heads: number;
  phase: number;
  seed: number;
  zeroTokens?: number;
  tinyTokens?: number;
}

const name = ({ width, height, heads, phase }: WindowCase): string =>
  `${width}x${height} heads ${heads} phase ${phase}`;

function inputs({ width, height, heads, seed, zeroTokens, tinyTokens }: WindowCase) {
  const options: QkvOptions = { seed, zeroTokens, tinyTokens };
  return {
    qkv: makeQkv(width * height, heads, options),
    prior: makePrior(heads, seed),
    scales: makeScales(heads, seed, 2, 6),
  };
}

/** Run our kernel; returns the whole `attended` allocation (padding rows included) and the kernel. */
async function runOurs(testCase: WindowCase, queries: WindowQueries = 32) {
  const { width, height, heads, phase } = testCase;
  const tokens = width * height;
  const data = inputs(testCase);
  const qkv = createTensor('qkv', tokens, heads * 96, 'f16');
  writeBuffer(qkv, data.qkv);
  const attended = createTensor('attended', tokens, heads * 32, 'e4');
  fillSentinel(attended);
  const k = createWindowAttention(
    { width, height, heads, phase, label: 'block 1' },
    { qkv, prior: createHalfVector(data.prior), scales: createF32Vector(data.scales), attended },
    { queries },
  );
  await runKernels(gpu.renderer, [k]);
  return { bytes: await readBuffer(gpu.renderer, attended, { validOnly: false }), kernel: k, data };
}

function expectOracle(testCase: WindowCase, bytes: Uint8Array, data: ReturnType<typeof inputs>): void {
  const { width, height, heads, phase } = testCase;
  const [shiftX, shiftY] = windowPhase(phase);
  const expected = oracleWindowAttention({ width, height, heads, shiftX, shiftY, ...data });
  const result = diffArrays(bytes.subarray(0, expected.length), expected);
  expect(result.mismatches, describeMismatches('attended', result, 2)).toBe(0);
  // R6: rows past width * height are never written.
  expect(bytes.subarray(expected.length).every((b) => b === SENTINEL_BYTE)).toBe(true);
}

// Every phase; edge windows that are partly outside the field (20x12, 44x40 and every shifted phase); heads 1, 2, 16.
const CASES: WindowCase[] = [
  { width: 20, height: 12, heads: 1, phase: 0, seed: 1 },
  { width: 20, height: 12, heads: 2, phase: 1, seed: 2 },
  { width: 20, height: 12, heads: 1, phase: 2, seed: 3 },
  { width: 20, height: 12, heads: 2, phase: 3, seed: 4 },
  { width: 44, height: 40, heads: 1, phase: 0, seed: 5 },
  { width: 44, height: 40, heads: 2, phase: 1, seed: 6 },
  { width: 44, height: 40, heads: 1, phase: 2, seed: 7 },
  { width: 44, height: 40, heads: 1, phase: 3, seed: 8 },
  { width: 24, height: 16, heads: 16, phase: 1, seed: 9 },
  { width: 24, height: 16, heads: 16, phase: 2, seed: 10 },
];

// Many all-zero tokens (norm = 1/sqrt(0) = inf, k * inf = 0 * inf = NaN -> published +0) and tiny tokens (the
// half square sum underflows or the norm overflows half).
const EDGE_CASES: WindowCase[] = [
  { width: 24, height: 16, heads: 2, phase: 1, seed: 21, zeroTokens: 0.3, tinyTokens: 0 },
  { width: 24, height: 16, heads: 2, phase: 3, seed: 22, zeroTokens: 0, tinyTokens: 0.4 },
  { width: 20, height: 12, heads: 1, phase: 0, seed: 23, zeroTokens: 0.25, tinyTokens: 0.25 },
];

describe('window attention vs the oracle of the composed reference', () => {
  it.for(CASES)('$width x $height, heads $heads, phase $phase', async (testCase) => {
    const { bytes, data } = await runOurs(testCase);
    expectOracle(testCase, bytes, data);
  });

  it.for(EDGE_CASES)('zero / tiny tokens: $width x $height, heads $heads, phase $phase', async (testCase) => {
    const { bytes, data } = await runOurs(testCase);
    expectOracle(testCase, bytes, data);
  });

  it.for([CASES[1], CASES[5], EDGE_CASES[0]])(
    '16 and 32 queries per workgroup give identical bytes: %#',
    async (testCase) => {
      const wide = await runOurs(testCase, 32);
      const narrow = await runOurs(testCase, 16);
      expect(narrow.kernel.workgroupSize).toEqual([256, 1, 1]);
      expect(wide.kernel.workgroupSize).toEqual([512, 1, 1]);
      const result = diffArrays(narrow.bytes, wide.bytes);
      expect(result.mismatches, describeMismatches(`16 vs 32 queries, ${name(testCase)}`, result, 2)).toBe(0);
    },
  );
});

describe('window attention kernel', () => {
  it('dispatch and workgroup memory match the reference', () => {
    // graph.js windowAttention: [heads, min(tasks, 65535), ceil(tasks / 65535)], tasks = windows * 64 / 32.
    expect(windowAttentionDispatch({ width: 20, height: 12, heads: 2, phase: 1 })).toEqual([2, 3 * 2 * 2, 1]);
    expect(windowAttentionDispatch({ width: 576, height: 512, heads: 1, phase: 0 })).toEqual([1, 72 * 64 * 2, 1]);
    // 1920x1152 at the field: 69120 tasks fold into z.
    expect(windowAttentionDispatch({ width: 1920, height: 1152, heads: 1, phase: 0 })).toEqual([1, 65535, 2]);
    expect(windowAttentionWorkgroupBytes(32)).toBe(24576);
    expect(windowAttentionWorkgroupBytes(16)).toBeLessThanOrEqual(24576);
  });

  it('generated WGSL: integer comparisons, no approximations, shared between same-shape dispatches', async () => {
    const a = await runOurs(CASES[0]);
    const b = await runOurs({ ...CASES[0], seed: 99 });
    const wgsl = kernelWGSL(gpu.renderer, a.kernel);
    expectIntegerComparisons(wgsl);
    expectNoApproximations(wgsl);
    expect(wgsl).toBe(kernelWGSL(gpu.renderer, b.kernel));
    expect(wgsl).toContain('@workgroup_size( 512, 1, 1 )');
    expect(a.kernel.label).toBe('block 1 attend');
    expect(a.kernel.kind).toBe('window_attend');
  });
});

describe('window attention vs the reference (needs shader-f16)', () => {
  it.for([...CASES, ...EDGE_CASES])(
    '$width x $height, heads $heads, phase $phase (seed $seed)',
    async (testCase, context) => {
      const ref = await RefKernels.create(gpu.device);
      const reason = ref.unavailableReason('window_attend');
      if (reason) {
        ref.destroy();
        context.skip(reason);
        return;
      }
      const { width, height, heads, phase } = testCase;
      const tokens = width * height;
      const data = inputs(testCase);
      const qkv = ref.tensor('qkv', tokens, heads * 96, 'f16', data.qkv);
      const attended = ref.tensor('attended', tokens, heads * 32, 'e4');
      ref.fill(attended, SENTINEL_BYTE);
      ref.windowAttention({
        qkv,
        attended,
        prior: ref.buffer(data.prior, 'prior'),
        scales: ref.buffer(data.scales, 'scales'),
        width,
        height,
        heads,
        phase,
        label: 'block 1',
      });
      await ref.run();
      const expected = await ref.read(attended, { validOnly: false });
      ref.destroy();

      // The oracle against the reference, then ours against the reference (whole allocations, sentinel included).
      const [shiftX, shiftY] = windowPhase(phase);
      const oracle = oracleWindowAttention({ width, height, heads, shiftX, shiftY, ...data });
      const vsOracle = diffArrays(oracle, expected.subarray(0, oracle.length));
      expect(vsOracle.mismatches, describeMismatches('oracle vs reference', vsOracle, 2)).toBe(0);
      const { bytes } = await runOurs(testCase);
      const vsOurs = diffArrays(bytes, expected);
      expect(vsOurs.mismatches, describeMismatches('ours vs reference', vsOurs, 2)).toBe(0);
    },
  );
});
