// The elementwise kernels (port of ops.wgsl) are byte-identical to the reference's ops.wgsl, run in Node on the same
// device, and to the CPU oracle - on inputs full of signed zeros, subnormals, infinities and NaN codes, at the graph's
// shapes, with odd level sizes, and above 2^24 bytes (local only).

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import * as oracle from '../numerics/oracle.js';
import { createHalfVector, createTensor, readBuffer, writeBuffer } from '../tensors.js';
import { runKernels, kernelWGSL } from '../tsl/KernelBuilder.js';
import type { NRTensor, TensorFormat } from '../types.js';
import { bothHalfNaN, describeMismatches, diffArrays, fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import {
  oracleConvertF32ToF16,
  oracleDownsample,
  oraclePostBlend,
  oracleUpsampleResidual,
  type OracleLevels,
} from '../../test/oracle/ops.js';
import { RefKernels, type RefTensor } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons, expectNoApproximations, normalizeWGSL } from '../../test/wgsl.js';
import {
  createConvertF32ToF16,
  createCopyWords,
  createDownsample,
  createPostBlend,
  createUpsampleResidual,
} from './ops.js';

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

const rng = new oracle.Xorshift(0x51f15e);

/** Half patterns of every class: ~5% signed zeros, subnormals, a few infinities and NaNs, the rest in [2^-10, 2^5]. */
const randomHalves = (count: number, { specials = true } = {}): Uint16Array =>
  Uint16Array.from({ length: count }, () => {
    const draw = rng.next();
    const sign = draw & 0x8000;
    const kind = (draw >>> 16) % 100;
    if (kind < 5) return sign;
    if (kind < 10) return sign | ((draw >>> 4) & 0x3ff);
    if (specials && kind === 10) return sign | 0x7c00;
    if (specials && kind === 11) return sign | 0x7c00 | (((draw >>> 4) & 0x3ff) | 1);
    return sign | ((5 + ((draw >>> 20) % 16)) << 10) | ((draw >>> 4) & 0x3ff);
  });

/** E4 codes of every class (both NaN codes and both zeros included). */
const randomE4 = (count: number): Uint8Array =>
  Uint8Array.from({ length: count }, () => {
    const draw = rng.next();
    if (draw % 17 === 0) return draw & 0x80;
    if (draw % 31 === 0) return (draw & 0x80) | 0x7f;
    return (draw >>> 8) & 0xff;
  });

/** Scales: positive and negative halves of moderate size, a few zeros. */
const randomScales = (count: number): Uint16Array =>
  Uint16Array.from({ length: count }, () => {
    const draw = rng.next();
    if (draw % 23 === 0) return draw & 0x8000;
    return (draw & 0x8000) | ((10 + ((draw >>> 20) % 8)) << 10) | ((draw >>> 4) & 0x3ff);
  });

/** One of our tensors initialized with `data`. */
function ours(label: string, rows: number, channels: number, format: TensorFormat, data?: ArrayBufferView): NRTensor {
  const tensor = createTensor(label, rows, channels, format);
  if (data) writeBuffer(tensor, data);
  return tensor;
}

/** A sentinel-filled output tensor on both sides. */
function outputs(label: string, rows: number, channels: number, format: TensorFormat): [NRTensor, RefTensor] {
  const mine = ours(label, rows, channels, format);
  fillSentinel(mine);
  const theirs = ref.tensor(label, rows, channels, format);
  ref.fill(theirs, SENTINEL_BYTE);
  return [mine, theirs];
}

/**
 * Whether the reference's own ops.wgsl loses the sign of a zero on this device. FXC (D3D12 without DXC, the local
 * default on Windows) does - the reference's `round_f16` of a tiny negative value publishes +0 there - while Vulkan
 * (NVIDIA, lavapipe) keeps it, as the WGSL says. Our TSL keeps it on both. Probed once with a pool whose exact result
 * is -0 (-2^-24 * 0.25 rounds to -0 on the half grid).
 */
let referenceDropsSignedZeros: boolean | undefined;
async function probeReferenceSignedZeros(): Promise<boolean> {
  if (referenceDropsSignedZeros !== undefined) return referenceDropsSignedZeros;
  const input = new Uint16Array(4 * 4).fill(0x8000);
  input.fill(0x8001, 0, 4); // pixel (0,0): -2^-24 in every channel, the rest -0
  const inF16 = ref.tensor('signed zero probe in', 4, 4, 'f16', input);
  const outE4 = ref.tensor('signed zero probe out', 1, 4, 'e4');
  ref.op('downsample', {
    count: 4,
    channels: 4,
    inWidth: 2,
    inHeight: 2,
    outWidth: 1,
    outHeight: 1,
    inF16,
    outE4,
    label: 'probe',
  });
  await ref.run();
  const word = new Uint32Array((await ref.read(outE4)).buffer)[0];
  referenceDropsSignedZeros = word !== 0x80808080;
  if (referenceDropsSignedZeros) {
    console.info('[ops] the reference ops.wgsl drops signed zeros on this device (FXC); comparing -0 vs +0 leniently');
  }
  return referenceDropsSignedZeros;
}

/**
 * Compare the whole allocation of both sides (padding rows must keep the sentinel too) value by value, after checking
 * the valid values against the oracle strictly. Where the reference loses signed zeros on this device (FXC), a -0 of
 * ours against a +0 of the reference is accepted - ours already matched the oracle there.
 */
async function expectSame(what: string, mine: NRTensor, theirs: RefTensor, expected?: ArrayLike<number>) {
  const bytes = await readBuffer(gpu.renderer, mine, { validOnly: false });
  const reference = await ref.read(theirs, { validOnly: false });
  const half = mine.format === 'f16';
  const view = (data: Uint8Array) => (half ? new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2) : data);
  const actual = view(bytes);
  if (expected) {
    const r2 = diffArrays(actual.subarray(0, expected.length), expected, {
      equivalent: half ? bothHalfNaN : undefined,
    });
    expect(r2.mismatches, describeMismatches(`${what} vs oracle`, r2, half ? 4 : 2)).toBe(0);
  }
  const negativeZero = half ? 0x8000 : 0x80;
  const lenient = await probeReferenceSignedZeros();
  const r = diffArrays(actual, view(reference), {
    equivalent: lenient ? (a, e) => a === negativeZero && e === 0 : undefined,
  });
  expect(r.mismatches, describeMismatches(`${what} vs reference`, r, half ? 4 : 2)).toBe(0);
}

describe('convert_f32_to_f16', () => {
  it('matches the reference and the oracle over every f32 exponent class', async () => {
    const rows = 100;
    const values = new Float32Array(rows * 16);
    const bits = new Uint32Array(values.buffer);
    for (let i = 0; i < bits.length; ++i) {
      const draw = rng.next();
      // Exponents 90..150 (half subnormals, normals, overflow) plus zeros, infinities, NaNs and f32 subnormals.
      const kind = draw % 40;
      bits[i] =
        kind === 0
          ? draw & 0x80000000
          : kind === 1
            ? (draw & 0x80000000) | 0x7f800000
            : kind === 2
              ? (draw & 0x80000000) | 0x7fc00000
              : ((draw & 0x807fffff) | ((90 + ((draw >>> 8) % 61)) << 23)) >>> 0;
    }
    const inMine = ours('features', rows, 16, 'f32', values);
    const [outMine, outRef] = outputs('features f16', rows, 16, 'f16');
    const inRef = ref.tensor('features', rows, 16, 'f32', values);
    ref.op('convert_f32_to_f16', { count: rows * 16, channels: 16, inF32: inRef, outF16: outRef, label: 'features' });
    await ref.run();
    const k = createConvertF32ToF16(
      { count: rows * 16, label: 'features to half' },
      { input: inMine, output: outMine },
    );
    await runKernels(gpu.renderer, [k]);
    await expectSame('convert_f32_to_f16', outMine, outRef, oracleConvertF32ToF16(values));
    const wgsl = kernelWGSL(gpu.renderer, k);
    expectIntegerComparisons(wgsl);
    expectNoApproximations(wgsl);
  });
});

describe('downsample', () => {
  it.for([
    { channels: 32, inWidth: 24, inHeight: 20, outWidth: 12, outHeight: 10 },
    { channels: 64, inWidth: 25, inHeight: 21, outWidth: 13, outHeight: 11 },
    { channels: 512, inWidth: 6, inHeight: 4, outWidth: 3, outHeight: 2 },
  ])('$inWidth x $inHeight -> $outWidth x $outHeight x $channels', async (levels) => {
    const { channels, inWidth, inHeight, outWidth, outHeight } = levels;
    const input = randomHalves(inWidth * inHeight * channels);
    const inMine = ours('pool in', inWidth * inHeight, channels, 'f16', input);
    const inRef = ref.tensor('pool in', inWidth * inHeight, channels, 'f16', input);
    const [outMine, outRef] = outputs('pool out', outWidth * outHeight, channels, 'e4');
    const count = outWidth * outHeight * channels;
    ref.op('downsample', {
      count,
      channels,
      inWidth,
      inHeight,
      outWidth,
      outHeight,
      inF16: inRef,
      outE4: outRef,
      label: 'pool',
    });
    await ref.run();
    const k = createDownsample({ ...levels, label: 'pool' }, { input: inMine, output: outMine });
    await runKernels(gpu.renderer, [k]);
    await expectSame('downsample', outMine, outRef, oracleDownsample(levels, input).e4);
    const wgsl = kernelWGSL(gpu.renderer, k);
    expectIntegerComparisons(wgsl);
    expectNoApproximations(wgsl);
  });

  it.for([{ outWidth: 1024, outHeight: 520 }])(
    'above 2^24 output bytes and 65535 workgroups: $outWidth x $outHeight x 32',
    async ({ outWidth, outHeight }, context) => {
      if (process.env.CI) return context.skip('large shape (~150 MB): local only');
      const channels = 32;
      const [inWidth, inHeight] = [outWidth * 2, outHeight * 2];
      const levels: OracleLevels = { channels, inWidth, inHeight, outWidth, outHeight };
      const count = outWidth * outHeight * channels;
      expect(count).toBeGreaterThan(2 ** 24);
      const input = randomHalves(inWidth * inHeight * channels, { specials: false });
      const inMine = ours('big pool in', inWidth * inHeight, channels, 'f16', input);
      const inRef = ref.tensor('big pool in', inWidth * inHeight, channels, 'f16', input);
      const [outMine, outRef] = outputs('big pool out', outWidth * outHeight, channels, 'e4');
      ref.op('downsample', {
        count,
        channels,
        inWidth,
        inHeight,
        outWidth,
        outHeight,
        inF16: inRef,
        outE4: outRef,
        label: 'pool',
      });
      await ref.run();
      const k = createDownsample({ ...levels, label: 'big pool' }, { input: inMine, output: outMine });
      expect(k.dispatch[1]).toBeGreaterThan(1);
      await runKernels(gpu.renderer, [k]);
      await expectSame('downsample (large)', outMine, outRef, oracleDownsample(levels, input).e4);
    },
  );
});

describe('upsample_residual', () => {
  it.for([
    { channels: 32, inWidth: 12, inHeight: 10, outWidth: 24, outHeight: 20, dual: true, scaleOffset: 0 },
    { channels: 64, inWidth: 12, inHeight: 10, outWidth: 24, outHeight: 20, dual: false, scaleOffset: 8 },
    { channels: 512, inWidth: 2, inHeight: 2, outWidth: 4, outHeight: 3, dual: false, scaleOffset: 0 },
  ])('$inWidth x $inHeight -> $outWidth x $outHeight x $channels, dual $dual', async (spec) => {
    const { channels, inWidth, inHeight, outWidth, outHeight, dual, scaleOffset } = spec;
    const rows = outWidth * outHeight;
    const count = rows * channels;
    const input = randomHalves(inWidth * inHeight * channels);
    const skip = randomE4(count);
    const scales = randomScales(channels + scaleOffset);
    const [outMine, outRef] = outputs('merged', rows, channels, 'e4');
    const [rawMine, rawRef] = outputs('merged raw', rows, channels, 'f16');
    ref.op('upsample_residual', {
      count,
      channels,
      inWidth,
      inHeight,
      outWidth,
      outHeight,
      auxA: scaleOffset,
      dual,
      inF16: ref.tensor('low', inWidth * inHeight, channels, 'f16', input),
      skipE4: ref.tensor('skip', rows, channels, 'e4', skip),
      aux: ref.buffer(scales),
      outE4: outRef,
      outF16: dual ? rawRef : undefined,
      label: 'merge',
    });
    await ref.run();
    const k = createUpsampleResidual(
      { channels, inWidth, inHeight, outWidth, outHeight, scaleOffset, label: 'block 39 merge' },
      {
        input: ours('low', inWidth * inHeight, channels, 'f16', input),
        skip: ours('skip', rows, channels, 'e4', skip),
        scale: createHalfVector(scales),
        output: outMine,
        outputF16: dual ? rawMine : undefined,
      },
    );
    await runKernels(gpu.renderer, [k]);
    const expected = oracleUpsampleResidual(spec, input, skip, scales, scaleOffset);
    await expectSame('upsample_residual e4', outMine, outRef, expected.e4);
    if (dual) await expectSame('upsample_residual f16', rawMine, rawRef, expected.f16);
    expectIntegerComparisons(kernelWGSL(gpu.renderer, k));
  });
});

describe('post_blend', () => {
  it.for([
    { channels: 32, inWidth: 12, inHeight: 10, outWidth: 24, outHeight: 20 },
    { channels: 32, inWidth: 13, inHeight: 11, outWidth: 26, outHeight: 21 },
  ])('$inWidth x $inHeight -> $outWidth x $outHeight x $channels, dual', async (levels) => {
    const { channels, inWidth, inHeight, outWidth, outHeight } = levels;
    const rows = outWidth * outHeight;
    const count = rows * channels;
    const input = randomE4(inWidth * inHeight * channels);
    const skip = randomE4(count);
    const scales = randomScales(channels * 2);
    const [outMine, outRef] = outputs('post merge', rows, channels, 'e4');
    const [rawMine, rawRef] = outputs('post merge raw', rows, channels, 'f16');
    ref.op('post_blend', {
      count,
      channels,
      inWidth,
      inHeight,
      outWidth,
      outHeight,
      auxA: 0,
      auxB: channels,
      dual: true,
      inE4: ref.tensor('low', inWidth * inHeight, channels, 'e4', input),
      skipE4: ref.tensor('block0', rows, channels, 'e4', skip),
      aux: ref.buffer(scales),
      outE4: outRef,
      outF16: rawRef,
      label: 'post blend',
    });
    await ref.run();
    const k = createPostBlend(
      { ...levels, label: 'post blend' },
      {
        input: ours('low', inWidth * inHeight, channels, 'e4', input),
        skip: ours('block0', rows, channels, 'e4', skip),
        scales: createHalfVector(scales),
        output: outMine,
        outputF16: rawMine,
      },
    );
    await runKernels(gpu.renderer, [k]);
    const expected = oraclePostBlend(levels, input, skip, scales);
    await expectSame('post_blend e4', outMine, outRef, expected.e4);
    await expectSame('post_blend f16', rawMine, rawRef, expected.f16);
  });
});

describe('copy_words', () => {
  it('copies a tensor inside the pass, after the kernel that wrote it', async () => {
    const [inWidth, inHeight, channels] = [8, 6, 32];
    const input = randomHalves(inWidth * inHeight * channels, { specials: false });
    const pooled = ours('pooled', 12, channels, 'e4');
    const capture = ours('boundary pooled', 12, channels, 'e4');
    fillSentinel(capture);
    const pool = createDownsample(
      { channels, inWidth, inHeight, outWidth: 4, outHeight: 3, label: 'pool' },
      { input: ours('in', inWidth * inHeight, channels, 'f16', input), output: pooled },
    );
    const copy = createCopyWords({ label: 'capture pooled' }, { source: pooled, target: capture });
    // The pool runs again after the copy on other data: the capture must hold the first result.
    const later = createDownsample(
      { channels, inWidth, inHeight, outWidth: 4, outHeight: 3, label: 'pool again' },
      { input: ours('zeros', inWidth * inHeight, channels, 'f16'), output: pooled },
    );
    await runKernels(gpu.renderer, [pool, copy, later]);
    const captured = await readBuffer(gpu.renderer, capture, { validOnly: false });
    const expected = new Uint8Array(capture.byteLength);
    expected.set(oracleDownsample({ channels, inWidth, inHeight, outWidth: 4, outHeight: 3 }, input).e4);
    const r = diffArrays(captured, expected);
    expect(r.mismatches, describeMismatches('capture', r, 2)).toBe(0);
    expect((await readBuffer(gpu.renderer, pooled)).every((b) => b === 0)).toBe(true);
  });
});

/**
 * Normalized WGSL with the helper functions sorted: three emits layout functions in the order its node cache first met
 * them, which depends on what was built before in the same renderer.
 */
function stableWGSL(wgsl: string): string {
  const [head, ...functions] = normalizeWGSL(wgsl).split(/\n(?=fn )/);
  const main = functions.findIndex((block) => block.startsWith('fn main'));
  const [entry] = functions.splice(main, 1);
  return [head, ...functions.toSorted(), entry].join('\n');
}

/** A small tensor for WGSL snapshots. */
const t = (format: TensorFormat, rows = 64, channels = 32) => ours(`snap ${format}`, rows, channels, format);

describe('generated WGSL', () => {
  it('one snapshot per kernel kind (catches codegen drift on a three bump)', () => {
    const levels = { channels: 32, inWidth: 8, inHeight: 8, outWidth: 4, outHeight: 4 };
    const kernels = [
      createConvertF32ToF16(
        { count: 64 * 16, label: 'convert' },
        { input: t('f32', 64, 16), output: t('f16', 64, 16) },
      ),
      createDownsample({ ...levels, label: 'pool' }, { input: t('f16'), output: t('e4') }),
      createUpsampleResidual(
        { ...levels, inWidth: 4, inHeight: 4, outWidth: 8, outHeight: 8, label: 'merge' },
        {
          input: t('f16'),
          skip: t('e4'),
          scale: createHalfVector(new Uint16Array(32)),
          output: t('e4'),
          outputF16: t('f16'),
        },
      ),
      createPostBlend(
        { ...levels, inWidth: 4, inHeight: 4, outWidth: 8, outHeight: 8, label: 'post blend' },
        { input: t('e4'), skip: t('e4'), scales: createHalfVector(new Uint16Array(64)), output: t('e4') },
      ),
      createCopyWords({ label: 'capture' }, { source: t('e4'), target: t('e4') }),
    ];
    for (const k of kernels) expect(stableWGSL(kernelWGSL(gpu.renderer, k))).toMatchSnapshot(k.kind);
  });
});
