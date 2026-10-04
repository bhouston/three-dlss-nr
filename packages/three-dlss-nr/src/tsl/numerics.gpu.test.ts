// Every TSL numerics helper, exhaustively on the GPU, against the reference's numerics fixture
// (web/fixtures/numerics.bin, produced by the Vulkan implementation), the reference's own WGSL running on the same
// device (shaders/numerics.wgsl + selftest.wgsl), and the TS oracle.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { floatBitsToUint, int, uint, uintBitsToFloat } from 'three/tsl';

import { numericsCases, readFixture } from '@ref/numerics_cases.js';

import { bothF32NaN, bothHalfNaN, describeMismatches, diffArrays } from '../../test/compare.js';
import { createGpuTestContext, runPerElement, wordsBuffer, type GpuTestContext } from '../../test/gpu.js';
import { referenceFile, RefSelftest } from '../../test/reference/refKernels.js';
import * as oracle from '../numerics/oracle.js';
import {
  nrDecodeE4m3,
  nrE4m3Exponent,
  nrEncodeE4m3,
  nrExpWeight,
  nrF13Cover,
  nrF13Start,
  nrF16Bits,
  nrF16Exponent,
  nrF16ToF32,
  nrFdpa16,
  nrFdpaF16x8,
  nrFixedToF16,
  nrMpCubicSilu,
  nrNormalExponent,
  nrPow2,
  nrPublishE4CodeGemm,
  nrRoundF16,
  nrRoundShiftRightEven,
  nrVitExpWeight,
} from './numerics.js';
import type { BufferSource } from './KernelBuilder.js';
import { loadE4, loadF32, loadHalf, loadHalfBits, packHalfPair, packWord4, u, type TSLNode } from './packed.js';

const ALL_HALVES = 65536;
const halfIndices = (): Uint32Array => Uint32Array.from({ length: ALL_HALVES }, (_, index) => index);
const nonFiniteHalf = (bits: number): boolean => (bits & 0x7c00) === 0x7c00;

/** Random f32 bit patterns, with every exponent class represented (zeros, subnormals, normals, inf, NaN). */
function randomF32Patterns(count: number, seed: number): Uint32Array {
  const rng = new oracle.Xorshift(seed);
  const words = new Uint32Array(count);
  for (let index = 0; index < count; ++index) {
    let bits = rng.next();
    // A quarter of the draws land near the half range, where the rounding paths are.
    if (index % 4 === 0) bits = (bits & 0x807fffff) | ((100 + (rng.next() % 50)) << 23);
    if (index % 64 === 1) bits &= 0x807fffff; // f32 subnormals and zeros
    words[index] = bits >>> 0;
  }
  words.set([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x477fe000, 0x477ff000, 0x33000000, 0x33000001]);
  return words;
}

let gpu: GpuTestContext & { dispose(): void };
let selftest: RefSelftest;
let cases: any[];

beforeAll(async () => {
  gpu = await createGpuTestContext();
  selftest = await RefSelftest.create(gpu.device);
  cases = numericsCases(readFixture(referenceFile('web/fixtures/numerics.bin')));
});

afterAll(() => gpu?.dispose());

const fixtureCase = (entryPoint: string): any => {
  const entry = cases.find((c) => c.entryPoint === entryPoint);
  if (!entry) throw new Error(`no fixture case ${entryPoint}`);
  return entry;
};

/**
 * One fixture case: our TSL on the GPU must equal the fixture (half NaN patterns compare equal to each other, as the
 * reference's own `compare` does). The reference's WGSL then runs on the same device as a cross-check of the device:
 * on D3D12 without DXC (FXC) it deviates by itself - FXC fails to compile its `bitcast<u32>(abs(x))` (E_FAIL) and does
 * not preserve some signed zeros - so its deviations are reported, not asserted.
 */
async function checkFixtureCase(
  entryPoint: string,
  compute: (index: TSLNode, inputs: Record<string, TSLNode>) => TSLNode,
  mask: number,
): Promise<void> {
  const entry = fixtureCase(entryPoint);
  const count: number = entry.count;
  const inputs: Record<string, BufferSource> = entry.gpuInputs ? { inputs: wordsBuffer(entry.gpuInputs) } : {};
  const ours = await runPerElement(gpu.renderer, { label: entryPoint, count, inputs, compute });
  const expected = Array.from({ length: count }, (_, index) => entry.expected[index] >>> 0);
  const options = {
    skip: entry.skipInput as ((i: number) => boolean) | undefined,
    equivalent: entry.kind === 'half' ? bothHalfNaN : undefined,
  };
  const masked = (words: Uint32Array) => Array.from(words, (w) => (w & mask) >>> 0);
  const vsFixture = diffArrays(masked(ours), expected, options);
  expect(vsFixture.mismatches, describeMismatches(`TSL ${entryPoint} vs fixture`, vsFixture)).toBe(0);

  let note: string;
  try {
    const reference = await selftest.run(entryPoint, count, entry.gpuInputs);
    note = describeMismatches(
      `reference WGSL ${entryPoint} on this device vs fixture`,
      diffArrays(masked(reference), expected, options),
    );
  } catch (error) {
    note = `reference WGSL ${entryPoint} does not run on this device: ${(error as Error).message.split('\n')[0]}`;
  }
  if (!note.endsWith(' equal')) console.warn(`[device note] ${note}`);
}

describe('TSL numerics vs the reference fixture (and the reference WGSL on this device)', () => {
  it('nrF16Bits over the F16B f32 patterns', async () => {
    await checkFixtureCase(
      'case_f16_bits',
      (index, { inputs }) => nrF16Bits(uintBitsToFloat(inputs.element(index))),
      0xffff,
    );
  });

  it('nrEncodeE4m3 over every half (E4EN)', async () => {
    await checkFixtureCase('case_e4m3_encode', (index) => nrEncodeE4m3(index), 0xff);
  });

  it('nrDecodeE4m3 over every byte (E4DE, including -0 for 0x80 and both NaN codes)', async () => {
    await checkFixtureCase('case_e4m3_decode', (index) => floatBitsToUint(nrDecodeE4m3(index)), 0xffffffff);
  });

  it('nrMpCubicSilu over every finite half (SILU)', async () => {
    await checkFixtureCase('case_silu', (index) => nrF16Bits(nrMpCubicSilu(nrF16ToF32(index))), 0xffff);
  });

  it('nrExpWeight over every finite half (EXPW)', async () => {
    await checkFixtureCase('case_exp_weight', (index) => nrF16Bits(nrExpWeight(nrF16ToF32(index))), 0xffff);
  });

  it('nrFdpa16 over the FDP8 cases (16 E4M3 products + half accumulator)', async () => {
    await checkFixtureCase(
      'case_fdpa_fp8',
      (index, { inputs }) => {
        const base = index.mul(u(33)).toVar();
        const a = Array.from({ length: 16 }, (_, k) => uintBitsToFloat(inputs.element(base.add(u(k)))).toVar());
        const b = Array.from({ length: 16 }, (_, k) => uintBitsToFloat(inputs.element(base.add(u(16 + k)))).toVar());
        return nrF16Bits(nrFdpa16(a, b, uintBitsToFloat(inputs.element(base.add(u(32))))));
      },
      0xffff,
    );
  });

  it('nrFdpaF16x8 over the FD16 cases (8 half products + half accumulator)', async () => {
    await checkFixtureCase(
      'case_fdpa_f16',
      (index, { inputs }) => {
        const base = index.mul(u(17)).toVar();
        const a = Array.from({ length: 8 }, (_, k) => uintBitsToFloat(inputs.element(base.add(u(k)))).toVar());
        const b = Array.from({ length: 8 }, (_, k) => uintBitsToFloat(inputs.element(base.add(u(8 + k)))).toVar());
        return nrF16Bits(nrFdpaF16x8(a, b, uintBitsToFloat(inputs.element(base.add(u(16))))));
      },
      0xffff,
    );
  });

  it('nrVitExpWeight over every finite half (against the oracle, as the reference does)', async () => {
    await checkFixtureCase('case_vit_exp_weight', (index) => nrF16Bits(nrVitExpWeight(nrF16ToF32(index))), 0xffff);
  });
});

describe('TSL numerics vs the TS oracle, exhaustive', () => {
  it('nrF16ToF32 over every half keeps payloads, signs and subnormals', async () => {
    const ours = await runPerElement(gpu.renderer, {
      label: 'f16_to_f32',
      count: ALL_HALVES,
      compute: (index) => floatBitsToUint(nrF16ToF32(index)),
    });
    const expected = Array.from(halfIndices(), (bits) => oracle.f16ToF32Bits(bits));
    const result = diffArrays(ours, expected);
    expect(result.mismatches, describeMismatches('nrF16ToF32', result)).toBe(0);
  });

  it('loadHalf (unpack2x16float) and loadHalfBits over every half', async () => {
    const halves = new Uint16Array(ALL_HALVES).map((_, index) => index);
    const buffer = { attribute: wordsBuffer(new Uint32Array(halves.buffer)).attribute };
    const bits = await runPerElement(gpu.renderer, {
      label: 'load_half_bits',
      count: ALL_HALVES,
      inputs: { halves: buffer },
      compute: (index, { halves: view }) => loadHalfBits(view, index),
    });
    expect(Array.from(bits)).toEqual(Array.from(halfIndices()));
    const values = await runPerElement(gpu.renderer, {
      label: 'load_half',
      count: ALL_HALVES,
      inputs: { halves: buffer },
      compute: (index, { halves: view }) => floatBitsToUint(loadHalf(view, index)),
    });
    const expected = Array.from(halfIndices(), (h) => oracle.f16ToF32Bits(h));
    const result = diffArrays(values, expected, { equivalent: bothF32NaN });
    expect(result.mismatches, describeMismatches('loadHalf', result)).toBe(0);
  });

  it('half rounding at every positive and negative halfway point and its f32 neighbours', async () => {
    const patterns: number[] = [];
    const addMidpoint = (midpoint: number) => {
      const bits = oracle.f32Bits(midpoint);
      for (const offset of [-1, 0, 1]) {
        patterns.push((bits + offset) >>> 0, ((bits + offset) | 0x80000000) >>> 0);
      }
    };
    for (let half = 0; half < 0x7bff; ++half) {
      addMidpoint((oracle.f16ToNumber(half) + oracle.f16ToNumber(half + 1)) / 2);
    }
    addMidpoint(65520); // finite-to-infinity tie, outside the direct f32 rounding path
    const words = Uint32Array.from(patterns);
    for (const [label, helper, expected] of [
      ['half_midpoint_bits', nrF16Bits, (value: number) => oracle.f16Bits(value)],
      [
        'half_midpoint_round',
        (value: TSLNode) => floatBitsToUint(nrRoundF16(value)),
        (value: number) => oracle.f32Bits(oracle.roundF16(value)),
      ],
    ] as const) {
      const actual = await runPerElement(gpu.renderer, {
        label,
        count: words.length,
        inputs: { words: wordsBuffer(words) },
        compute: (index, { words: view }) => helper(loadF32(view, index)),
      });
      const reference = Array.from(words, (bits) => expected(oracle.f32FromBits(bits)));
      const diff = diffArrays(actual, reference);
      expect(diff.mismatches, describeMismatches(label, diff)).toBe(0);
    }
  });

  it('nrF16Bits and nrRoundF16 over 2^20 random f32 patterns', async () => {
    const words = randomF32Patterns(1 << 20, 0x1234567);
    const bits = await runPerElement(gpu.renderer, {
      label: 'f16_bits_random',
      count: words.length,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) => nrF16Bits(loadF32(view, index)),
    });
    const expectedBits = Array.from(words, (w) => oracle.f16Bits(oracle.f32FromBits(w)));
    const resultBits = diffArrays(bits, expectedBits, { equivalent: bothHalfNaN });
    expect(resultBits.mismatches, describeMismatches('nrF16Bits', resultBits)).toBe(0);
    const rounded = await runPerElement(gpu.renderer, {
      label: 'round_f16_random',
      count: words.length,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) => floatBitsToUint(nrRoundF16(loadF32(view, index))),
    });
    const expectedRounded = Array.from(words, (w) => oracle.f32Bits(oracle.roundF16(oracle.f32FromBits(w))));
    const resultRounded = diffArrays(rounded, expectedRounded, { equivalent: bothF32NaN });
    expect(resultRounded.mismatches, describeMismatches('nrRoundF16', resultRounded)).toBe(0);
  });

  it('nrPublishE4CodeGemm over every half value and 2^20 random f32 patterns (-0 -> 0x00)', async () => {
    const halves = await runPerElement(gpu.renderer, {
      label: 'publish_e4_halves',
      count: ALL_HALVES,
      compute: (index) => nrPublishE4CodeGemm(nrF16ToF32(index)),
    });
    const expectedHalves = Array.from(halfIndices(), (h) => oracle.publishE4CodeGemm(oracle.f16ToNumber(h)));
    const resultHalves = diffArrays(halves, expectedHalves);
    expect(resultHalves.mismatches, describeMismatches('publish (halves)', resultHalves)).toBe(0);
    expect(halves[0x8000]).toBe(0);

    const words = randomF32Patterns(1 << 20, 0xbeef);
    const ours = await runPerElement(gpu.renderer, {
      label: 'publish_e4_random',
      count: words.length,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) => nrPublishE4CodeGemm(loadF32(view, index)),
    });
    const expected = Array.from(words, (w) => oracle.publishE4CodeGemm(oracle.f32FromBits(w)));
    // f32 subnormal inputs are skipped: backends may flush them to zero (D3D12 does, R10), which turns a tiny negative
    // value's 0x80 into 0x00. The network never forms f32 subnormals (every published value is a half).
    const f32Subnormal = (index: number) => (words[index] & 0x7f800000) === 0 && (words[index] & 0x7fffff) !== 0;
    const result = diffArrays(ours, expected, { skip: f32Subnormal });
    expect(result.mismatches, describeMismatches('publish (f32)', result)).toBe(0);
  });

  it('nrRoundShiftRightEven over random values and every shift 0..40', async () => {
    const rng = new oracle.Xorshift(0x51f7);
    const count = 41 * 4096;
    const words = new Uint32Array(count * 2);
    for (let index = 0; index < count; ++index) {
      const value = rng.next();
      const shift = index % 41;
      // Exercise exact halfway remainders too.
      words[index * 2] = index % 3 === 0 && shift > 0 && shift < 32 ? (value | (2 ** (shift - 1))) >>> 0 : value;
      words[index * 2 + 1] = shift;
    }
    const ours = await runPerElement(gpu.renderer, {
      label: 'round_shift_right_even',
      count,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) =>
        nrRoundShiftRightEven(view.element(index.mul(u(2))), view.element(index.mul(u(2)).add(u(1)))),
    });
    const expected = Array.from({ length: count }, (_, index) =>
      oracle.roundShiftRightEven(words[index * 2], words[index * 2 + 1]),
    );
    const result = diffArrays(ours, expected);
    expect(result.mismatches, describeMismatches('nrRoundShiftRightEven', result)).toBe(0);
  });

  it('nrPow2 for every exponent -192..191 (clamped to the normal range)', async () => {
    const ours = await runPerElement(gpu.renderer, {
      label: 'pow2',
      count: 384,
      compute: (index) => floatBitsToUint(nrPow2(int(index).sub(int(192)))),
    });
    const expected = Array.from({ length: 384 }, (_, index) =>
      oracle.f32Bits(2 ** Math.min(Math.max(index - 192, -126), 127)),
    );
    expect(Array.from(ours)).toEqual(expected);
  });

  it('nrNormalExponent, nrE4m3Exponent, nrF16Exponent over every half', async () => {
    for (const [name, helper, reference] of [
      ['normal', nrNormalExponent, oracle.normalExponent],
      ['e4m3', nrE4m3Exponent, oracle.e4m3Exponent],
      ['f16', nrF16Exponent, oracle.f16Exponent],
    ] as const) {
      const ours = await runPerElement(gpu.renderer, {
        label: `${name}_exponent`,
        count: ALL_HALVES,
        compute: (index) => uint(helper(nrF16ToF32(index))),
      });
      const expected = Array.from(halfIndices(), (h) =>
        nonFiniteHalf(h) ? 128 >>> 0 : reference(oracle.f16ToNumber(h)) >>> 0,
      );
      const result = diffArrays(ours, expected, { skip: (h) => nonFiniteHalf(h) });
      expect(result.mismatches, describeMismatches(`${name} exponent`, result)).toBe(0);
    }
  });

  it('nrF13Start and nrF13Cover over every half / every E4 pair', async () => {
    const starts = await runPerElement(gpu.renderer, {
      label: 'f13_start',
      count: ALL_HALVES,
      compute: (index) => uint(nrF13Start(nrF16ToF32(index))),
    });
    const expectedStarts = Array.from(halfIndices(), (h) => oracle.f13Start(oracle.f16ToNumber(h)) >>> 0);
    const r1 = diffArrays(starts, expectedStarts, { skip: (h) => nonFiniteHalf(h) });
    expect(r1.mismatches, describeMismatches('nrF13Start', r1)).toBe(0);
    const covers = await runPerElement(gpu.renderer, {
      label: 'f13_cover',
      count: ALL_HALVES,
      compute: (index) =>
        uint(nrF13Cover(int(-21), nrDecodeE4m3(index.bitAnd(u(0xff))), nrDecodeE4m3(index.shiftRight(u(8))))),
    });
    const expectedCovers = Array.from(
      halfIndices(),
      (index) => oracle.f13Cover(-21, oracle.e4m3ToNumber(index & 0xff), oracle.e4m3ToNumber(index >> 8)) >>> 0,
    );
    const r2 = diffArrays(covers, expectedCovers);
    expect(r2.mismatches, describeMismatches('nrF13Cover', r2)).toBe(0);
  });

  it('nrFixedToF16 over 2^18 random (sum, exponent) pairs', async () => {
    const rng = new oracle.Xorshift(0xfeed);
    const count = 1 << 18;
    const words = new Uint32Array(count * 2);
    for (let index = 0; index < count; ++index) {
      const magnitudeBits = 1 + (rng.next() % 31); // up to 2^31 - 1
      let sum = rng.next() % 2 ** magnitudeBits;
      if (rng.next() & 1) sum = -sum;
      words[index * 2] = sum >>> 0;
      words[index * 2 + 1] = ((rng.next() % 80) - 64) >>> 0; // binary exponent -64..15
    }
    const ours = await runPerElement(gpu.renderer, {
      label: 'fixed_to_f16',
      count,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) =>
        nrF16Bits(nrFixedToF16(int(view.element(index.mul(u(2)))), int(view.element(index.mul(u(2)).add(u(1)))))),
    });
    const expected = Array.from({ length: count }, (_, index) =>
      oracle.f16Bits(oracle.fixedToF16(words[index * 2] | 0, words[index * 2 + 1] | 0)),
    );
    const result = diffArrays(ours, expected);
    expect(result.mismatches, describeMismatches('nrFixedToF16', result)).toBe(0);
  });

  it('nrDecodeE4m3 -> nrEncodeE4m3 round-trips every code (NaN codes -> signed zero)', async () => {
    const ours = await runPerElement(gpu.renderer, {
      label: 'e4_roundtrip',
      count: 256,
      compute: (index) => nrEncodeE4m3(nrF16Bits(nrDecodeE4m3(index))),
    });
    const expected = Array.from({ length: 256 }, (_, code) => ((code & 0x7f) === 0x7f ? code & 0x80 : code));
    expect(Array.from(ours)).toEqual(expected);
  });
});

describe('packed access', () => {
  it('loadE4 reads every byte position; packWord4 / packHalfPair rebuild words', async () => {
    const words = randomF32Patterns(4096, 0xabc);
    const bytes = new Uint8Array(words.buffer);
    const read = await runPerElement(gpu.renderer, {
      label: 'load_e4',
      count: bytes.length,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) => loadE4(view, index),
    });
    expect(Array.from(read)).toEqual(Array.from(bytes));
    const repacked = await runPerElement(gpu.renderer, {
      label: 'pack_word4',
      count: words.length,
      inputs: { words: wordsBuffer(words) },
      compute: (index, { words: view }) => {
        const at = (lane: number) => loadE4(view, index.mul(u(4)).add(u(lane)));
        const halves = packHalfPair(loadHalfBits(view, index.mul(u(2))), loadHalfBits(view, index.mul(u(2)).add(u(1))));
        // Both rebuilds must equal the word; return their xor with it (zero when equal) or-ed together.
        return packWord4(at(0), at(1), at(2), at(3))
          .bitXor(view.element(index))
          .bitOr(halves.bitXor(view.element(index)));
      },
    });
    expect(repacked.every((word) => word === 0)).toBe(true);
  });
});
