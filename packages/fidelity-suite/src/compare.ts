// Byte comparison of raw tensors for the parity suite's exactness.json.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41).

/** The verdict on one tensor of one renderer against the reference. */
export interface TensorVerdict {
  verdict: 'bit-exact' | 'differs';
  /** Bytes compared (the reference's length). */
  bytes: number;
  /** Differing bytes, plus the length difference when the lengths differ. */
  mismatches: number;
  /** First differing byte, -1 when bit-exact. */
  first: number;
}

/** Compare `actual` with `expected` byte for byte. A length difference is never bit-exact. */
export function compareBytes(actual: Uint8Array, expected: Uint8Array): TensorVerdict {
  let mismatches = Math.abs(actual.length - expected.length);
  let first = mismatches ? Math.min(actual.length, expected.length) : -1;
  const length = Math.min(actual.length, expected.length);
  for (let i = 0; i < length; ++i) {
    if (actual[i] === expected[i]) continue;
    if (first < 0 || i < first) first = i;
    mismatches += 1;
  }
  return { verdict: mismatches ? 'differs' : 'bit-exact', bytes: expected.length, mismatches, first };
}

/** The bytes of a typed array view. */
export const bytesOf = (view: ArrayBufferView): Uint8Array =>
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
