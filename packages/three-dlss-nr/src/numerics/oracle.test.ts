// The TS oracle against the reference's numerics fixture (all seven sections, produced by the Vulkan implementation
// of OpenDLSS-NR) and against the reference port's own JavaScript (ports/browser-webgpu/src/numerics.js, model.js,
// numerics_cases.js).

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import * as ref from '@ref/numerics.js';
import * as refModel from '@ref/model.js';
import { numericsCases, readFixture, Xorshift as RefXorshift } from '@ref/numerics_cases.js';

import * as oracle from './oracle.js';

const fixture = readFixture(
  new Uint8Array(
    readFileSync(
      new URL('../../../../reference/OpenDLSS-NR/ports/browser-webgpu/web/fixtures/numerics.bin', import.meta.url),
    ),
  ),
);
const cases: any[] = numericsCases(fixture);
const caseFor = (entryPoint: string): any => cases.find((c) => c.entryPoint === entryPoint);

const halfNaN = (bits: number) => (bits & 0x7c00) === 0x7c00 && (bits & 0x3ff) !== 0;
const nonFinite = (bits: number) => (bits & 0x7c00) === 0x7c00;
const same = (x: number, y: number) => Object.is(x, y) || (Number.isNaN(x) && Number.isNaN(y));

/** Count mismatches of `actual(i)` against the case's expected values, the way the reference's `compare` does. */
function mismatches(entry: any, actual: (i: number) => number): number {
  let count = 0;
  for (let i = 0; i < entry.count; ++i) {
    if (entry.skipInput?.(i)) continue;
    const e = entry.expected[i];
    const a = actual(i);
    if (a === e) continue;
    if (entry.kind === 'half' && halfNaN(e) && halfNaN(a)) continue;
    count += 1;
  }
  return count;
}

describe('oracle vs the numerics fixture (Vulkan reference)', () => {
  it('has all seven sections', () => {
    expect([...fixture.keys()].toSorted()).toEqual(['E4DE', 'E4EN', 'EXPW', 'F16B', 'FD16', 'FDP8', 'SILU']);
  });

  it('F16B: f16Bits over arbitrary f32 patterns', () => {
    const entry = caseFor('case_f16_bits');
    expect(entry.count).toBe(32768);
    expect(mismatches(entry, (i) => oracle.f16Bits(oracle.f32FromBits(entry.gpuInputs[i])))).toBe(0);
  });

  it('E4EN: e4m3FromF16Bits over every half', () => {
    expect(mismatches(caseFor('case_e4m3_encode'), (i) => oracle.e4m3FromF16Bits(i))).toBe(0);
  });

  it('E4DE: e4m3ToNumber over every byte (bit patterns, -0 included)', () => {
    expect(mismatches(caseFor('case_e4m3_decode'), (i) => oracle.f32Bits(oracle.e4m3ToNumber(i)))).toBe(0);
  });

  it('SILU: mpCubicSilu over every finite half', () => {
    const entry = caseFor('case_silu');
    expect(mismatches(entry, (i) => oracle.f16Bits(oracle.mpCubicSilu(oracle.f16ToNumber(i))))).toBe(0);
  });

  it('EXPW: expWeight over every finite half', () => {
    const entry = caseFor('case_exp_weight');
    expect(mismatches(entry, (i) => oracle.f16Bits(oracle.expWeight(oracle.f16ToNumber(i))))).toBe(0);
  });

  it('FDP8: adaFp8Fdpa16 over random E4M3 operands', () => {
    const entry = caseFor('case_fdpa_fp8');
    const { a, b, accumulators } = entry.operands;
    const actual = (c: number) =>
      oracle.f16Bits(
        oracle.adaFp8Fdpa16(a.subarray(c * 16, c * 16 + 16), b.subarray(c * 16, c * 16 + 16), 16, accumulators[c]),
      );
    expect(entry.count).toBe(16384);
    expect(mismatches(entry, actual)).toBe(0);
  });

  it('FD16: adaF16Fdpa8 over random half operands', () => {
    const entry = caseFor('case_fdpa_f16');
    const { a, b, accumulators } = entry.operands;
    const actual = (c: number) =>
      oracle.f16Bits(
        oracle.adaF16Fdpa8(a.subarray(c * 8, c * 8 + 8), b.subarray(c * 8, c * 8 + 8), 8, accumulators[c]),
      );
    expect(mismatches(entry, actual)).toBe(0);
  });
});

describe('oracle vs the reference JavaScript', () => {
  it('every half: conversions, publications, activations, exponents', () => {
    let differences = 0;
    for (let h = 0; h < 65536; ++h) {
      const value = oracle.f16ToNumber(h);
      if (!same(value, ref.f16ToNumber(h))) differences += 1;
      if (oracle.e4m3FromF16Bits(h) !== ref.e4m3FromF16Bits(h)) differences += 1;
      if (!same(oracle.roundF16(value), ref.roundF16(value))) differences += 1;
      if (!nonFinite(h)) {
        if (!same(oracle.mpCubicSilu(value), ref.mpCubicSilu(value))) differences += 1;
        if (!same(oracle.expWeight(value), ref.expWeight(value))) differences += 1;
        if (!same(oracle.vitExpWeight(value), ref.vitExpWeight(value))) differences += 1;
        if (oracle.normalExponent(value) !== ref.normalExponent(value)) differences += 1;
        if (oracle.f13Start(value) !== ref.f13Start(value)) differences += 1;
      }
    }
    expect(differences).toBe(0);
  });

  it('every byte decodes alike; f13Cover over every pair of codes', () => {
    for (let byte = 0; byte < 256; ++byte) {
      expect(Object.is(oracle.e4m3ToNumber(byte), ref.e4m3ToNumber(byte))).toBe(true);
    }
    for (let x = 0; x < 256; ++x) {
      for (let y = 0; y < 256; ++y) {
        const [a, b] = [oracle.e4m3ToNumber(x), oracle.e4m3ToNumber(y)];
        if (oracle.f13Cover(-21, a, b) !== ref.f13Cover(-21, a, b)) throw new Error(`f13Cover ${x} ${y}`);
      }
    }
  });

  it('random f32 patterns: f16Bits, e4m3FromNumber; random shifts and fixed-point sums', () => {
    const rng = new oracle.Xorshift(0xc0ffee);
    for (let i = 0; i < 200_000; ++i) {
      const value = oracle.f32FromBits(rng.next());
      expect(oracle.f16Bits(value)).toBe(ref.f16Bits(value));
      expect(oracle.e4m3FromNumber(value)).toBe(ref.e4m3FromNumber(value));
      const word = rng.next();
      const shift = rng.next() % 40;
      expect(oracle.roundShiftRightEven(word, shift)).toBe(ref.roundShiftRightEven(word, shift));
      let sum = rng.next() % 2 ** (1 + (rng.next() % 31));
      if (rng.next() & 1) sum = -sum;
      const exponent = (rng.next() % 80) - 64;
      expect(Object.is(oracle.fixedToF16(sum, exponent), ref.fixedToF16(sum, exponent))).toBe(true);
    }
  });

  it('siluTable equals the reference', () => {
    expect(oracle.siluTable()).toEqual(ref.siluTable());
  });

  it('Xorshift draws the same sequence', () => {
    const ours = new oracle.Xorshift(0x9e3779b9);
    const theirs = new RefXorshift(0x9e3779b9);
    for (let i = 0; i < 1000; ++i) expect(ours.next()).toBe(theirs.next());
  });

  it('index maps equal model.js and invert correctly', () => {
    for (let k = 0; k < 4096; ++k) {
      expect(oracle.packedInputIndex(k)).toBe(refModel.packedInputIndex(k));
      expect(oracle.inversePackedInputIndex(oracle.packedInputIndex(k))).toBe(k);
      expect(oracle.inversePackedInputIndex(k)).toBe(refModel.inversePackedInputIndex(k));
      // The swizzle stays inside a 16-product group.
      expect(oracle.packedInputIndex(k) >> 4).toBe(k >> 4);
    }
    for (let token = 0; token < 64; ++token) {
      expect(oracle.tiledToken(token)).toBe(refModel.tiledToken(token));
      expect(oracle.inverseTiledToken(token)).toBe(refModel.inverseTiledToken(token));
      expect(oracle.inverseTiledToken(oracle.tiledToken(token))).toBe(token);
    }
    for (const [K, N] of [
      [32, 32],
      [128, 96],
      [64, 256],
      [1024, 48],
    ]) {
      const seen = new Uint8Array(K * N);
      for (let k = 0; k < K; ++k) {
        for (let n = 0; n < N; ++n) {
          const index = oracle.packedWeightIndex(k, n, N);
          expect(index).toBe(refModel.packedWeightIndex(k, n, N));
          seen[index] += 1;
        }
      }
      expect(seen.every((count) => count === 1)).toBe(true);
    }
    // packedF16WeightIndex is private in model.js; check it is a bijection on the two shapes the network uses.
    for (const [K, N] of [
      [16, 32],
      [32, 4],
    ]) {
      const tiles = Math.ceil(N / 16);
      const seen = new Uint8Array(K * tiles * 16);
      for (let k = 0; k < K; ++k) for (let n = 0; n < N; ++n) seen[oracle.packedF16WeightIndex(k, n, N)] += 1;
      expect(seen.every((count) => count <= 1)).toBe(true);
    }
  });
});

describe('production publications (src/matmul) vs the numerics.js grid', () => {
  it('publishE4CodeGemm equals e4m3FromF16Bits on every half except -0 (0x00, not 0x80)', () => {
    for (let h = 0; h < 65536; ++h) {
      const expected = h === 0x8000 ? 0 : oracle.e4m3FromF16Bits(h);
      if (oracle.publishE4CodeGemm(oracle.f16ToNumber(h)) !== expected) throw new Error(`half 0x${h.toString(16)}`);
    }
  });

  it('exactE4Code(fp8DomainBitQuant(h)) equals e4m3FromF16Bits on every half but -0 (keeps 0x80 for tiny negatives)', () => {
    for (let h = 0; h < 65536; ++h) {
      const code = oracle.exactE4Code(oracle.fp8DomainBitQuant(oracle.f16ToNumber(h)));
      // fp8_domain returns +0 for a zero input, so -0 publishes 0x00; a nonzero value rounding to zero keeps its sign.
      const expected = h === 0x8000 ? 0 : oracle.e4m3FromF16Bits(h);
      if (code !== expected) throw new Error(`half 0x${h.toString(16)}: ${code}`);
    }
    expect(oracle.exactE4Code(oracle.fp8DomainBitQuant(-1e-6))).toBe(0x80);
    expect(oracle.publishE4CodeGemm(-1e-6)).toBe(0x80);
    expect(oracle.publishE4CodeGemm(-0)).toBe(0);
  });

  it('siluE4CodeTable is the E4 publication of siluTable (a -0 activation publishes 0x00)', () => {
    const codes = oracle.siluE4CodeTable();
    const halves = oracle.siluTable();
    for (let h = 0; h < 65536; ++h) {
      const expected = halves[h] === 0x8000 ? 0 : oracle.e4m3FromF16Bits(halves[h]);
      if (codes[h] !== expected) throw new Error(`half 0x${h.toString(16)}`);
    }
    expect(codes[0x8000]).toBe(0);
    expect(codes[0x8001]).toBe(0x80); // silu(-2^-24) is a tiny negative: rounds to zero, keeps the sign
  });

  it('roundHalfEven is WGSL round', () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6, 7.5].map(oracle.roundHalfEven)).toEqual([0, 2, 2, -0, -2, 2, 3, 8]);
  });
});
