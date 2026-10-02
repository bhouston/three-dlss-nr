// Bit-level comparisons for tests, built on the reference's own verdict functions.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). `compareCodes` / `compareFloats` are the
// reference's (ports/browser-webgpu/src/parity.js): byte-for-byte E4M3 codes with signed zeros counted apart, and
// bitwise f32. The rest are small helpers with readable failure messages.

import { compareCodes, compareFloats } from '@ref/parity.js';

import type { NRTensor } from '../src/types.js';
import { fillBuffer } from '../src/tensors.js';

export { compareCodes, compareFloats };

/** The byte tests prefill every output with, on both sides (R6: an unwritten output stays 0xCD, not zero). */
export const SENTINEL_BYTE = 0xcd;

/** Prefill one of our tensors (or any buffer) with the sentinel. */
export function fillSentinel(source: Pick<NRTensor, 'attribute'>): void {
  fillBuffer(source, SENTINEL_BYTE);
}

/** Outcome of an element-wise comparison. */
export interface Mismatches {
  count: number;
  mismatches: number;
  /** Up to `limit` first differences. */
  first: { index: number; actual: number; expected: number }[];
}

/**
 * Compare two integer arrays element by element. `skip(i)` excludes inputs, `equivalent(a, e)` accepts a pair that is
 * not identical (e.g. two NaN patterns).
 */
export function diffArrays(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  {
    skip,
    equivalent,
    limit = 8,
  }: { skip?: (i: number) => boolean; equivalent?: (a: number, e: number) => boolean; limit?: number } = {},
): Mismatches {
  if (actual.length !== expected.length) {
    throw new Error(`cannot compare ${actual.length} values with ${expected.length}`);
  }
  const result: Mismatches = { count: expected.length, mismatches: 0, first: [] };
  for (let i = 0; i < expected.length; ++i) {
    if (skip?.(i)) continue;
    const a = actual[i];
    const e = expected[i];
    if (a === e || equivalent?.(a, e)) continue;
    result.mismatches += 1;
    if (result.first.length < limit) result.first.push({ index: i, actual: a, expected: e });
  }
  return result;
}

const hex = (value: number, digits: number): string => `0x${(value >>> 0).toString(16).padStart(digits, '0')}`;

/** A one-line description of a comparison, for assertion messages. */
export function describeMismatches(what: string, result: Mismatches, digits = 8): string {
  if (!result.mismatches) return `${what}: ${result.count} equal`;
  const examples = result.first
    .map(({ index, actual, expected }) => `[${index}] got ${hex(actual, digits)} want ${hex(expected, digits)}`)
    .join(', ');
  return `${what}: ${result.mismatches} of ${result.count} differ: ${examples}`;
}

/** True for two half NaN patterns (payloads and signs of NaNs are not compared). */
export const bothHalfNaN = (a: number, e: number): boolean =>
  (a & 0x7c00) === 0x7c00 && (a & 0x3ff) !== 0 && (e & 0x7c00) === 0x7c00 && (e & 0x3ff) !== 0;

/** True for two f32 NaN patterns. */
export const bothF32NaN = (a: number, e: number): boolean =>
  ((a >>> 0) & 0x7fffffff) > 0x7f800000 && ((e >>> 0) & 0x7fffffff) > 0x7f800000;
