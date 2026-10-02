// The ViT attention kernels (`vit_normalize`, `vit_attend`) against the CPU oracle (test/oracle/attention.ts) and, where
// the reference's vit.wgsl compiles (not under FXC; e.g. Dawn's Vulkan backend or lavapipe), against the reference.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { describeMismatches, diffArrays, fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { oracleVitAttend, oracleVitNormalize } from '../../test/oracle/attention.js';
import { makeQkv, makeScales } from '../../test/oracle/attentionInputs.js';
import { RefKernels } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons, expectNoApproximations } from '../../test/wgsl.js';
import { grid1d } from '../geometry.js';
import { createF32Vector, createTensor, readBuffer, writeBuffer } from '../tensors.js';
import { kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import type { VitSpec } from '../types.js';
import { createVitAttend, createVitNormalize } from './vit.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

const HEADS = 32; // the network's ViT: 1024 channels

interface VitCase {
  tokens: number;
  seed: number;
  zeroTokens?: number;
  tinyTokens?: number;
}

/** Inputs of one case: raw qkv halves and learned scales. */
function inputs({ tokens, seed, zeroTokens, tinyTokens }: VitCase) {
  return {
    qkv: makeQkv(tokens, HEADS, { seed, zeroTokens, tinyTokens }),
    scales: makeScales(HEADS, seed, 0.15, 0.5),
    paddedTokens: (tokens + 63) & ~63,
  };
}

/** Run our normalize + attend; returns the whole normalized allocation and the attended valid bytes. */
async function runOurs(spec: VitSpec, qkvHalves: Uint16Array, scaleValues: Float32Array) {
  const qkv = createTensor('vit qkv', spec.tokens, HEADS * 96, 'f16');
  writeBuffer(qkv, qkvHalves);
  // Padding rows of `vit normalized` must stay zero (the network's tensor is zero-filled and never written there).
  const normalized = createTensor('vit normalized', spec.paddedTokens, HEADS * 96, 'e4');
  const attended = createTensor('vit attended', spec.tokens, HEADS * 32, 'e4');
  fillSentinel(attended);
  const normalize = createVitNormalize(spec, { qkv, scales: createF32Vector(scaleValues), normalized });
  const attend = createVitAttend(spec, { normalized, attended });
  await runKernels(gpu.renderer, [normalize, attend]);
  return {
    normalized: await readBuffer(gpu.renderer, normalized, { validOnly: false }),
    attended: await readBuffer(gpu.renderer, attended, { validOnly: false }),
    kernels: [normalize, attend],
  };
}

const CASES: (VitCase & { name: string })[] = [
  { name: '64 tokens (no padding)', tokens: 64, seed: 11 },
  { name: '96 tokens padded to 128 (the 512x512 ViT)', tokens: 96, seed: 12 },
  {
    name: '72 tokens padded to 128, many zero and tiny tokens',
    tokens: 72,
    seed: 13,
    zeroTokens: 0.2,
    tinyTokens: 0.2,
  },
];

describe('ViT vs oracle', () => {
  it.for(CASES)('$name', async (testCase) => {
    const { qkv, scales, paddedTokens } = inputs(testCase);
    const spec: VitSpec = { tokens: testCase.tokens, heads: HEADS, paddedTokens, label: 'block 31' };
    const ours = await runOurs(spec, qkv, scales);

    const expectedNormalized = oracleVitNormalize({ qkv, scales, tokens: spec.tokens, heads: HEADS });
    const validNormalized = expectedNormalized.length;
    const normalizedDiff = diffArrays(ours.normalized.subarray(0, validNormalized), expectedNormalized);
    expect(normalizedDiff.mismatches, describeMismatches('vit normalized', normalizedDiff, 2)).toBe(0);
    expect(
      ours.normalized.subarray(validNormalized).every((b) => b === 0),
      'padding rows stay zero',
    ).toBe(true);

    const fullNormalized = new Uint8Array(paddedTokens * HEADS * 96);
    fullNormalized.set(expectedNormalized);
    const expectedAttended = oracleVitAttend({
      normalized: fullNormalized,
      tokens: spec.tokens,
      heads: HEADS,
      paddedTokens,
    });
    const attendedDiff = diffArrays(ours.attended.subarray(0, expectedAttended.length), expectedAttended);
    expect(attendedDiff.mismatches, describeMismatches('vit attended', attendedDiff, 2)).toBe(0);
    expect(ours.attended.subarray(expectedAttended.length).every((b) => b === SENTINEL_BYTE)).toBe(true);
  });

  it('generated WGSL: integer comparisons, no approximations, buffers-only differences share the text', async () => {
    const { qkv, scales } = inputs({ tokens: 96, seed: 1 });
    const spec: VitSpec = { tokens: 96, heads: HEADS, paddedTokens: 128, label: 'block 31' };
    const a = await runOurs(spec, qkv, scales);
    const b = await runOurs({ ...spec, label: 'block 32' }, qkv, scales);
    for (let k = 0; k < 2; ++k) {
      const wgsl = kernelWGSL(gpu.renderer, a.kernels[k]);
      expectIntegerComparisons(wgsl);
      expectNoApproximations(wgsl);
      expect(wgsl).toBe(kernelWGSL(gpu.renderer, b.kernels[k]));
    }
    expect(a.kernels[0].dispatch).toEqual(grid1d(96 * HEADS * 8));
    expect(a.kernels[1].dispatch).toEqual([HEADS, 96, 1]);
    expect(a.kernels.map((k) => k.label)).toEqual(['block 31 normalize', 'block 31 attend']);
  });
});

describe('ViT vs the reference vit.wgsl', () => {
  it.for(CASES)('$name', async (testCase, context) => {
    const { qkv, scales, paddedTokens } = inputs(testCase);
    const ref = await RefKernels.create(gpu.device, { paddedVitTokens: paddedTokens });
    const reason = ref.unavailableReason('vit_normalize') ?? ref.unavailableReason('vit_attend');
    if (reason) {
      ref.destroy();
      context.skip(reason);
      return;
    }
    const { tokens } = testCase;
    const channels = HEADS * 32;
    const refQkv = ref.tensor('vit qkv', tokens, channels * 3, 'f16', qkv);
    const refNormalized = ref.tensor('vit normalized', paddedTokens, channels * 3, 'e4');
    const refAttended = ref.tensor('vit attended', tokens, channels, 'e4');
    ref.fill(refAttended, SENTINEL_BYTE);
    const scaleBuffer = ref.buffer(scales, 'vit scales');
    const params = { tokens, heads: HEADS, channels, paddedTokens };
    ref.vit(
      'vit_normalize',
      { 1: refQkv.buffer, 2: scaleBuffer, 5: refNormalized.buffer },
      params,
      grid1d(tokens * HEADS * 8),
    );
    ref.vit('vit_attend', { 4: refNormalized.buffer, 6: refAttended.buffer }, params, [HEADS, tokens, 1]);
    await ref.run();
    const expectedNormalized = await ref.read(refNormalized, { validOnly: false });
    const expectedAttended = await ref.read(refAttended, { validOnly: false });
    ref.destroy();

    const ours = await runOurs({ tokens, heads: HEADS, paddedTokens, label: 'block 31' }, qkv, scales);
    const n = diffArrays(ours.normalized, expectedNormalized);
    expect(n.mismatches, describeMismatches('vit normalized', n, 2)).toBe(0);
    const a = diffArrays(ours.attended, expectedAttended);
    expect(a.mismatches, describeMismatches('vit attended', a, 2)).toBe(0);
  });
});
