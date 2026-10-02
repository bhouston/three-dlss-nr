// Port of OpenDLSS-NR ports/browser-webgpu/src/numerics.js (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/src/numerics.js), plus the pieces of
// the reference that the production kernels compose but that have no JS twin there:
//   * `publishE4CodeGemm`  - `publish_e4_code` of src/matmul/packed-activation.js (the GEMM's E4 publication),
//   * `fp8DomainBitQuant`  - `fp8_domain` as rewritten by src/matmul/bit-quant.js,
//   * `exactE4Code`        - `exact_e4_output_code` of src/matmul/packed-activation.js,
//   * `siluE4CodeTable`    - the 64 KiB E4-code SiLU table (src/matmul/silu-table.js + packed-activation.js),
//   * `packedInputIndex`, `inversePackedInputIndex`, `packedWeightIndex`, `packedF16WeightIndex`, `tiledToken`,
//     `inverseTiledToken` - src/model.js,
//   * `Xorshift` - src/numerics_cases.js.
//
// The network's publication grid, on the CPU. This module is the port's oracle: tests compare the TSL kernels
// against it and against the reference, so it is written for legibility rather than speed, and it never takes a
// shortcut the shaders cannot take. JavaScript doubles are wider than every precision used here, so each f32-level
// step goes through Math.fround (see the reference's docs/numerics.md).

const scratch = new DataView(new ArrayBuffer(4));

/** f32 bit pattern of a number. */
export function f32Bits(value: number): number {
  scratch.setFloat32(0, value);
  return scratch.getUint32(0);
}

/** Number from an f32 bit pattern. */
export function f32FromBits(bits: number): number {
  scratch.setUint32(0, bits >>> 0);
  return scratch.getFloat32(0);
}

/** Round an arbitrary double down to what an f32 would hold, which is where the WGSL side always starts. */
export const toF32 = (value: number): number => Math.fround(value);

/** Shift right with round-to-nearest-even, the rounding every narrowing conversion here uses. */
export function roundShiftRightEven(value: number, shift: number): number {
  if (shift === 0) return value >>> 0;
  if (shift > 31) return 0;
  const quotient = value >>> shift;
  const remainder = (value & ((1 << shift) - 1)) >>> 0;
  const halfway = 2 ** (shift - 1);
  return quotient + (remainder > halfway || (remainder === halfway && quotient & 1) ? 1 : 0);
}

/** IEEE binary16 bit pattern, round-to-nearest-even. */
export function f16Bits(value: number): number {
  const bits = f32Bits(value);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa ? 0x7e00 : 0x7c00);
  let halfExponent = exponent - 112;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    return sign | roundShiftRightEven(mantissa | 0x800000, 14 - halfExponent);
  }
  let rounded = roundShiftRightEven(mantissa, 13);
  if (rounded === 0x400) {
    rounded = 0;
    halfExponent += 1;
  }
  if (halfExponent >= 31) return sign | 0x7c00;
  return sign | (halfExponent << 10) | rounded;
}

/** Number from an IEEE binary16 bit pattern. */
export function f16ToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * f32 bit pattern of a half bit pattern, assembled on the bits so that NaN payloads and signs survive - what the
 * WGSL `f16_to_f32` (and our `nrF16ToF32`) returns. `f32Bits(f16ToNumber(h))` differs only for NaNs.
 */
export function f16ToF32Bits(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0x1f) return (sign | 0x7f800000 | (mantissa << 13)) >>> 0;
  return f32Bits(f16ToNumber(bits));
}

/** One publication to the half grid. */
export const roundF16 = (value: number): number => f16ToNumber(f16Bits(value));

/**
 * E4M3FN code of an already half-rounded value: RNE, finite saturation at 448, NaN -> +0. The sign of a zero
 * survives, as the hardware conversion leaves it: +0 and -0 decode alike, but the published bytes differ.
 */
export function e4m3FromF16Bits(half: number): number {
  if ((half & 0x7c00) === 0x7c00 && (half & 0x03ff) !== 0) return 0;
  if ((half & 0x7fff) === 0) return (half >>> 8) & 0x80;
  const negative = half & 0x8000 ? 0x80 : 0;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  let code: number;
  if (exponent === 31) {
    code = 0x7e;
  } else if (exponent <= 8) {
    // Below 2^-6 the E4M3 grid is subnormal: the half significand shifts down onto a fixed step of 2^-9.
    const significand = exponent === 0 ? mantissa : 1024 + mantissa;
    code = Math.min(roundShiftRightEven(significand, exponent === 0 ? 15 : 16 - exponent), 8);
  } else {
    let e4Exponent = exponent - 8;
    let e4Mantissa = roundShiftRightEven(mantissa, 7);
    if (e4Mantissa === 8) {
      e4Mantissa = 0;
      e4Exponent += 1;
    }
    code = e4Exponent > 15 || (e4Exponent === 15 && e4Mantissa > 6) ? 0x7e : (e4Exponent << 3) | e4Mantissa;
  }
  return negative | code;
}

export const e4m3FromNumber = (value: number): number => e4m3FromF16Bits(f16Bits(value));

/**
 * E4M3FN value of a byte. The one NaN code reads as a signed zero, which is what the shaders do with it.
 */
export function e4m3ToNumber(byte: number): number {
  const sign = byte & 0x80 ? -1 : 1;
  const exponent = (byte >>> 3) & 0xf;
  const mantissa = byte & 0x7;
  if (exponent === 0) return sign * mantissa * 2 ** -9;
  if (exponent === 0xf && mantissa === 0x7) return sign < 0 ? -0 : 0;
  return sign * (1 + mantissa / 8) * 2 ** (exponent - 7);
}

export const normalExponent = (value: number): number => ((f32Bits(Math.abs(value)) >>> 23) & 0xff) - 127;
export const e4m3Exponent = (value: number): number => Math.max(normalExponent(value), -6);
export const f16Exponent = (value: number): number => Math.max(normalExponent(value), -14);

// The tensor core's dot product, as fixed point: the terms are accumulated as exact integers in units of
// 2^(max - 13) so that the JS and the WGSL cannot drift apart through a difference in summation order or a fused
// multiply-add the shader compiler decided to emit.

/** The shared exponent of a step that starts from this accumulator; -21 is the empty value. */
export const f13Start = (accumulator: number): number => (accumulator !== 0 ? f16Exponent(accumulator) : -21);

/** Widen the shared exponent to cover one product; a zero operand is skipped, not clamped. */
export function f13Cover(maximumExponent: number, a: number, b: number): number {
  if (a === 0 || b === 0) return maximumExponent;
  return Math.max(maximumExponent, e4m3Exponent(a) + e4m3Exponent(b));
}

/** One term of the aligned sum, in units of 2^(max - 13). */
export const f13Term = (value: number, scale: number): number => Math.trunc(toF32(value * scale));

/** The aligned sum, published once to the half grid. */
export const f13Finish = (units: number, maximumExponent: number): number =>
  roundF16(units * 2 ** (maximumExponent - 13));

/**
 * Exact signed integer times 2^binaryExponent -> half. The f16 step needs this rather than a rounding of the sum,
 * because its fixed-point total carries more significand than a half holds.
 */
export function fixedToF16(fixedSum: number, binaryExponent: number): number {
  if (fixedSum === 0) return 0;
  const negative = fixedSum < 0;
  const magnitude = Math.abs(fixedSum);
  const msb = 31 - Math.clz32(magnitude);
  let valueExponent = msb + binaryExponent;
  let halfBits = negative ? 0x8000 : 0;
  if (valueExponent >= -14) {
    let significand = msb > 10 ? roundShiftRightEven(magnitude, msb - 10) : magnitude << (10 - msb);
    if (significand >= 2048) {
      significand = 1024;
      valueExponent += 1;
    }
    if (valueExponent >= 16) halfBits |= 0x7c00;
    else halfBits |= ((valueExponent + 15) << 10) | (significand - 1024);
  } else {
    const subnormalScale = binaryExponent + 24;
    const mantissa =
      subnormalScale >= 0 ? magnitude << subnormalScale : roundShiftRightEven(magnitude, -subnormalScale);
    halfBits |= Math.min(mantissa, 1024);
  }
  return f16ToNumber(halfBits);
}

/** Operand list for the FDPA helpers: anything indexable. */
export type Operands = ArrayLike<number>;

/** One half of a k32 FP8 step: 16 products plus the incoming accumulator, aligned, truncated, summed, rounded. */
export function adaFp8Fdpa16(a: Operands, b: Operands, count: number, accumulator: number): number {
  if (!Number.isFinite(accumulator)) return accumulator;
  let maximumExponent = f13Start(accumulator);
  for (let i = 0; i < count; ++i) maximumExponent = f13Cover(maximumExponent, a[i], b[i]);
  const scale = 2 ** (13 - maximumExponent);
  let units = f13Term(accumulator, scale);
  for (let i = 0; i < count; ++i) units += f13Term(toF32(a[i] * b[i]), scale);
  return f13Finish(units, maximumExponent);
}

/** The f16 step: 8 products against 24 fractional bits, for the chains that never narrow to E4M3. */
export function adaF16Fdpa8(a: Operands, b: Operands, count: number, accumulator: number): number {
  let maximumExponent = f13Start(accumulator);
  for (let i = 0; i < count; ++i) {
    if (a[i] !== 0 && b[i] !== 0) {
      maximumExponent = Math.max(maximumExponent, f16Exponent(a[i]) + f16Exponent(b[i]));
    }
  }
  const scale = 2 ** (24 - maximumExponent);
  let units = Math.trunc(toF32(accumulator * scale));
  for (let i = 0; i < count; ++i) units += Math.trunc(toF32(toF32(a[i] * b[i]) * scale));
  return fixedToF16(units, maximumExponent - 24);
}

/** MpCubicSiLU: five half publications, the two inner steps f32 fused multiply-adds with exact products. */
export function mpCubicSilu(value: number): number {
  const bounded = roundF16(Math.min(Math.max(value, -4), 4));
  const absolute = roundF16(Math.abs(bounded));
  const inner = roundF16(toF32(-0.055908203125 * absolute + 0.447265625));
  const polynomial = roundF16(toF32(bounded * inner + 0.89453125));
  return roundF16(toF32(value * polynomial));
}

/**
 * The window blocks' attention weight: an affine map on the score, dropped into the half exponent field. The clamp
 * is what keeps the shifted pattern a finite half, so it belongs to the function, not to safety.
 */
export function expWeight(score: number): number {
  const affine = roundF16(toF32(score * 0.044921875 + 1.30078125));
  const clamped = Math.min(Math.max(affine, 1.03125), 1.5693359375);
  return f16ToNumber((((f16Bits(clamped) << 5) >>> 0) + 0x8000) & 0xffff);
}

/**
 * The ViT's variant: a 4-bit shift instead of 5, a different bias, and an affine evaluated in halves. The constants
 * are the half values native holds (not the f32 spellings 0.08953946828842163, 1.7093614339828491).
 */
export function vitExpWeight(score: number): number {
  const affine = roundF16(score * 0.08953857421875 + 1.708984375);
  const clamped = Math.min(Math.max(affine, 1.439453125), 1.9775390625);
  return f16ToNumber((((f16Bits(clamped) << 4) >>> 0) + 0x4000) & 0xffff);
}

/** The SiLU lookup the fused kernels index by half bit pattern: half bits of `mpCubicSilu`. */
export function siluTable(): Uint16Array {
  const table = new Uint16Array(65536);
  for (let bits = 0; bits < 65536; ++bits) table[bits] = f16Bits(mpCubicSilu(f16ToNumber(bits)));
  return table;
}

// ---------------------------------------------------------------------------------------------------------------
// Production-only publications (the composed GEMM, src/matmul/*.js). WGSL `round` is round-half-to-even.
// ---------------------------------------------------------------------------------------------------------------

/** Round to the nearest integer, ties to even (WGSL `round`). */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  const rounded = fraction > 0.5 ? floor + 1 : fraction < 0.5 ? floor : floor % 2 === 0 ? floor : floor + 1;
  return rounded === 0 && (value < 0 || Object.is(value, -0)) ? -0 : rounded;
}

/**
 * The FP8 GEMM's E4 publication (`publish_e4_code`, src/matmul/packed-activation.js): NaN -> 0, saturate at 448
 * (0x7e), E4 subnormals `round(m * 512)`, normals RNE to three fraction bits directly on the f32 pattern. The sign
 * bit is set only for `value < 0`, so **-0 publishes as 0x00** (unlike `e4m3FromF16Bits`).
 */
export function publishE4CodeGemm(value: number): number {
  const v = toF32(value);
  if (Number.isNaN(v)) return 0;
  const magnitude = Math.min(Math.abs(v), 448);
  if (magnitude === 0) return 0;
  let code: number;
  if (magnitude < 0.015625) {
    code = roundHalfEven(toF32(magnitude * 512));
  } else {
    const bits = f32Bits(magnitude);
    const rounded = ((bits + 0x7ffff + ((bits >>> 20) & 1)) & 0xfff00000) >>> 0;
    code = (rounded >>> 20) - 960;
  }
  return code | (v < 0 ? 0x80 : 0);
}

/**
 * `fp8_domain` as src/matmul/bit-quant.js rewrites it: the E4M3 value as an f32, RNE, saturating at 448, NaN -> +0.
 * Returns `-quantized` for a negative input, so a small negative value that rounds to zero is **-0**; a zero input
 * (either sign) returns +0.
 */
export function fp8DomainBitQuant(value: number): number {
  const v = toF32(value);
  if (Number.isNaN(v)) return 0;
  const magnitude = Math.min(Math.abs(v), 448);
  if (magnitude === 0) return 0;
  let quantized: number;
  if (magnitude < 0.015625) {
    quantized = toF32(roundHalfEven(toF32(magnitude * 512)) / 512);
  } else {
    const bits = f32Bits(magnitude);
    quantized = f32FromBits(((bits + 0x7ffff + ((bits >>> 20) & 1)) & 0xfff00000) >>> 0);
  }
  return v < 0 ? -quantized : quantized;
}

/**
 * `exact_e4_output_code` (src/matmul/packed-activation.js): the code of a value that is already an exact finite E4M3
 * value (signed zero included; the sign is taken from the bit pattern, so -0 -> 0x80).
 */
export function exactE4Code(value: number): number {
  const v = toF32(value);
  const magnitude = Math.abs(v);
  const bits = f32Bits(magnitude);
  const code = magnitude >= 0.015625 ? (bits >>> 20) - 960 : Math.trunc(toF32(magnitude * 512));
  return (code | ((f32Bits(v) >>> 24) & 0x80)) >>> 0;
}

/**
 * The GEMM's SiLU E4-code table, indexed by the half bit pattern of the accumulator:
 * `exactE4Code(fp8DomainBitQuant(mpCubicSilu(h)))`: a zero activation publishes 0x00 (also -0), a tiny negative one
 * that rounds to zero 0x80. The reference builds it on the GPU with hardware f16 conversions
 * (`createSiluTable` + `createPackedSiluTable`); entries for NaN inputs are not specified by WGSL (min/max of NaN).
 */
export function siluE4CodeTable(): Uint8Array {
  const table = new Uint8Array(65536);
  for (let bits = 0; bits < 65536; ++bits) {
    table[bits] = exactE4Code(fp8DomainBitQuant(mpCubicSilu(f16ToNumber(bits))));
  }
  return table;
}

// ---------------------------------------------------------------------------------------------------------------
// Index maps (src/model.js).
// ---------------------------------------------------------------------------------------------------------------

/**
 * The within-32 activation index every FP8 GEMM's A operand is in (`native_chained_input_index`): logical input
 * channel `k` is stored at byte `packedInputIndex(k)` of the row. Rotates bits 1-3 and leaves bit 4 alone, so it stays
 * inside a 16-product group.
 */
export function packedInputIndex(k: number): number {
  const base = k & ~31;
  const within = k & 31;
  const half = within & 16;
  const quarter = within & 15;
  return base + half + (quarter >> 2) * 2 + (quarter & 1) + ((quarter & 2) !== 0 ? 8 : 0);
}

export function inversePackedInputIndex(k: number): number {
  const base = k & ~31;
  const within = k & 31;
  return base + (within & 17) + ((within & 2) << 1) + ((within & 4) << 1) + ((within & 8) >> 2);
}

/** Byte index of weight (k, n) inside a packed FP8 matrix of N columns, as the model file stores it. */
export function packedWeightIndex(k: number, n: number, outputChannels: number): number {
  const kTile = k >> 5;
  const kIn = k & 31;
  const nTile = n >> 7;
  const nIn = n & 127;
  const nHalf = nIn >> 6;
  const nGroup = (nIn & 63) >> 4;
  const nInGroup = nIn & 15;
  const lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
  const byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
  return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}

/**
 * Half index of an f16 weight inside a packed f16 matrix: 16x16 tiles in (k, n) order, each tile one m16n8k16 B
 * fragment pair (private `packedF16WeightIndex` of src/model.js).
 */
export function packedF16WeightIndex(inputChannel: number, outputChannel: number, outputChannels: number): number {
  const nTiles = Math.ceil(outputChannels / 16);
  const tile = (inputChannel >> 4) * nTiles + (outputChannel >> 4);
  const k = inputChannel & 15;
  const n = outputChannel & 15;
  const lane = ((n & 7) << 2) | ((k & 7) >> 1);
  const fragment = (k >= 8 ? 2 : 0) + (k & 1);
  return tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + fragment;
}

/** Natural window token (row-major in the 8x8 window) -> physical token (4x4 tiles of 16). */
export function tiledToken(token: number): number {
  const x = token & 7;
  const y = token >> 3;
  return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
}

/** Physical token -> natural window token. */
export function inverseTiledToken(token: number): number {
  const tile = token >> 4;
  const within = token & 15;
  const x = (tile & 1) * 4 + (within & 3);
  const y = (tile >> 1) * 4 + (within >> 2);
  return y * 8 + x;
}

// ---------------------------------------------------------------------------------------------------------------
// Deterministic inputs (src/numerics_cases.js).
// ---------------------------------------------------------------------------------------------------------------

/** The 32-bit xorshift the reference's numerics fixture draws its inputs from. */
export class Xorshift {
  state: number;
  constructor(seed: number) {
    this.state = seed >>> 0;
  }
  next(): number {
    let x = this.state;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.state = x;
    return x;
  }
}

/** A finite half pattern from a random draw (non-finite exponents lose their top exponent bit). */
export function finiteHalf(draw: number): number {
  let bits = draw & 0xffff;
  if ((bits & 0x7c00) === 0x7c00) bits &= 0x7bff;
  return bits;
}

/** True for a half NaN bit pattern. */
export const isHalfNaN = (bits: number): boolean => (bits & 0x7c00) === 0x7c00 && (bits & 0x03ff) !== 0;
