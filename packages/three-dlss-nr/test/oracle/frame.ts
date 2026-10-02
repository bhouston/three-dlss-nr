// Comparison helpers for the frame kernels (preprocess, input features, compose), which are not bit-gated.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). The frame uses the hardware's
// transcendentals (exp, pow, log2, cos, sin), so a port agrees with the reference's frame.wgsl to within rounding
// rather than by construction; these helpers measure how closely.

import { f16Bits, f32Bits } from '../../src/numerics/oracle.js';

/** Signed-magnitude half bits as an ordered integer (adjacent halves differ by 1; +0 and -0 are both 0). */
const orderedHalf = (bits: number): number => (bits & 0x8000 ? -(bits & 0x7fff) : bits & 0x7fff);

/** Distance in half ulps between two values that are on the half grid. */
export const halfUlps = (a: number, b: number): number => Math.abs(orderedHalf(f16Bits(a)) - orderedHalf(f16Bits(b)));

export interface FeatureReport {
  /** Per lane 3..15: every value bit-identical. */
  exactLanes: boolean[];
  /** Noise lanes 0-2: largest difference in half ulps, and how many values were not bit-identical. */
  noiseMaxHalfUlps: number;
  noiseMismatches: number;
  /** Lanes 3..15: largest relative difference |a - e| / max(|e|, 2^-14), for tolerance checks. */
  maxRelative: number;
  /** Lanes 3..15: values not bit-identical. */
  mismatches: number;
  summary: string;
}

/** Compare two f32 feature arrays `[rows][16]` lane by lane. */
export function compareFeatures(actual: Float32Array, expected: Float32Array, rows: number): FeatureReport {
  const exactLanes = Array.from({ length: 13 }, () => true);
  let noiseMaxHalfUlps = 0;
  let noiseMismatches = 0;
  let maxRelative = 0;
  let mismatches = 0;
  let first = '';
  for (let row = 0; row < rows; ++row) {
    for (let lane = 0; lane < 16; ++lane) {
      const index = row * 16 + lane;
      const [a, e] = [actual[index], expected[index]];
      const same = f32Bits(a) === f32Bits(e);
      if (lane < 3) {
        if (!same) {
          noiseMismatches += 1;
          noiseMaxHalfUlps = Math.max(noiseMaxHalfUlps, halfUlps(a, e));
        }
        continue;
      }
      if (same) continue;
      exactLanes[lane - 3] = false;
      mismatches += 1;
      maxRelative = Math.max(maxRelative, Math.abs(a - e) / Math.max(Math.abs(e), 2 ** -14));
      if (!first) first = `; first: row ${row} lane ${lane} got ${a} want ${e}`;
    }
  }
  const summary =
    `lanes 3-15: ${mismatches} of ${rows * 13} differ (max relative ${maxRelative.toExponential(2)})${first}; ` +
    `noise lanes: ${noiseMismatches} of ${rows * 3} differ (max ${noiseMaxHalfUlps} half ulps)`;
  return { exactLanes, noiseMaxHalfUlps, noiseMismatches, maxRelative, mismatches, summary };
}

/** Largest |a - e| / max(|e|, floor) over two arrays, with the count of non-identical values. */
export function relativeDifference(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  floor = 2 ** -14,
): { max: number; mismatches: number; first: string } {
  let max = 0;
  let mismatches = 0;
  let first = '';
  for (let i = 0; i < expected.length; ++i) {
    const [a, e] = [actual[i], expected[i]];
    if (Object.is(a, e) || (Number.isNaN(a) && Number.isNaN(e))) continue;
    mismatches += 1;
    const relative = Math.abs(a - e) / Math.max(Math.abs(e), floor);
    if (relative > max) {
      max = relative;
      first = `[${i}] got ${a} want ${e}`;
    }
  }
  return { max, mismatches, first };
}
