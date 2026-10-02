// CPU oracles of the elementwise kernels (reference shaders/ops.wgsl), for tests.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Literal transcriptions of ops.wgsl over
// the TS numerics oracle; ops.gpu.test.ts checks them against the reference on the GPU and against our TSL kernels.

import * as oracle from '../../src/numerics/oracle.js';

export interface OracleLevels {
  channels: number;
  inWidth: number;
  inHeight: number;
  outWidth: number;
  outHeight: number;
}

const e4 = (bytes: Uint8Array, index: number): number => oracle.e4m3ToNumber(bytes[index]);
const half = (halves: Uint16Array, index: number): number => oracle.f16ToNumber(halves[index]);

/** Published outputs of one op: E4 codes and the raw halves (what `dual` would write). */
export interface OracleOpResult {
  e4: Uint8Array;
  f16: Uint16Array;
}

function publishAll(values: Float64Array): OracleOpResult {
  const f16 = Uint16Array.from(values, (value) => oracle.f16Bits(value));
  return { e4: Uint8Array.from(f16, (bits) => oracle.e4m3FromF16Bits(bits)), f16 };
}

/** `convert_f32_to_f16`: half bits of each f32. */
export const oracleConvertF32ToF16 = (values: Float32Array): Uint16Array =>
  Uint16Array.from(values, (value) => oracle.f16Bits(value));

/** `downsample`: 2x2 pool of halves, E4 out; pixels whose 2x2 block leaves the input publish zeros. */
export function oracleDownsample(levels: OracleLevels, input: Uint16Array): OracleOpResult {
  const { channels, inWidth, inHeight, outWidth, outHeight } = levels;
  const values = new Float64Array(outWidth * outHeight * channels);
  for (let oy = 0; oy < outHeight; ++oy) {
    for (let ox = 0; ox < outWidth; ++ox) {
      const [sx, sy] = [ox * 2, oy * 2];
      if (sx + 1 >= inWidth || sy + 1 >= inHeight) continue;
      for (let c = 0; c < channels; ++c) {
        const at = (x: number, y: number) => half(input, (y * inWidth + x) * channels + c);
        const top = oracle.roundF16(at(sx, sy) + at(sx + 1, sy));
        const bottom = oracle.roundF16(at(sx, sy + 1) + at(sx + 1, sy + 1));
        values[(oy * outWidth + ox) * channels + c] = oracle.roundF16(oracle.roundF16(top + bottom) * 0.25);
      }
    }
  }
  return publishAll(values);
}

const sourceIndex = (levels: OracleLevels, pixel: number, c: number): number =>
  ((Math.floor(pixel / levels.outWidth) >> 1) * levels.inWidth + ((pixel % levels.outWidth) >> 1)) * levels.channels +
  c;

/** `upsample_residual`: `roundF16(up + skip * scale)`. */
export function oracleUpsampleResidual(
  levels: OracleLevels,
  input: Uint16Array,
  skip: Uint8Array,
  scale: Uint16Array,
  scaleOffset = 0,
): OracleOpResult {
  const { channels, outWidth, outHeight } = levels;
  const values = new Float64Array(outWidth * outHeight * channels);
  for (let index = 0; index < values.length; ++index) {
    const c = index % channels;
    const source = sourceIndex(levels, Math.floor(index / channels), c);
    // f32 arithmetic: the product is exact (E4 x half), the sum rounds once to f32, then to half.
    values[index] = oracle.roundF16(
      Math.fround(half(input, source) + Math.fround(e4(skip, index) * half(scale, scaleOffset + c))),
    );
  }
  return publishAll(values);
}

/** `post_blend`: `roundF16(roundF16(in * sA) + skip * sB)`. */
export function oraclePostBlend(
  levels: OracleLevels,
  input: Uint8Array,
  skip: Uint8Array,
  scales: Uint16Array,
  auxA = 0,
  auxB = levels.channels,
): OracleOpResult {
  const { channels, outWidth, outHeight } = levels;
  const values = new Float64Array(outWidth * outHeight * channels);
  for (let index = 0; index < values.length; ++index) {
    const c = index % channels;
    const source = sourceIndex(levels, Math.floor(index / channels), c);
    const upsampled = oracle.roundF16(Math.fround(e4(input, source) * half(scales, auxA + c)));
    values[index] = oracle.roundF16(Math.fround(upsampled + Math.fround(e4(skip, index) * half(scales, auxB + c))));
  }
  return publishAll(values);
}
