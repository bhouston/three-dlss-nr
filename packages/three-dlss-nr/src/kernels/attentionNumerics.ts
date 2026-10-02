// Short exact forms of the two publications the attention kernels run most often.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The reference spells `round_f16` as
// `f16_to_f32(f16_bits(x))` (shaders/numerics.wgsl) and `publish_fp8` as `decode_e4m3(encode_e4m3(f16_bits(x)))`
// (src/window/base.js); both are pure functions of the f32 input, and the forms below compute the same f32 for every
// f32 input with a handful of integer operations instead of the branch-free bit assembly of `nrRoundF16` (whose
// inlined copies dominate FXC's compile time and the kernels' run time). They are checked against the TS oracle
// (`roundF16`, `e4m3ToNumber(e4m3FromF16Bits(f16Bits(x)))`) exhaustively over every half and every half midpoint
// neighbourhood, and over random f32 patterns (attentionNumerics.gpu.test.ts).
//
// Both rely on WGSL `round` being round-half-to-even (as `nrPublishE4CodeGemm` does) and use it only on exact
// products by powers of two; no other floating-point arithmetic happens.

import { Fn, floatBitsToUint, min, round, uintBitsToFloat } from 'three/tsl';

import { nrEncodeE4m3, nrF16Bits } from '../tsl/numerics.js';
import { f, pick, u, type TSLNode } from '../tsl/packed.js';

/**
 * `round_f16(x)`: x rounded to the nearest half (ties to even), as an f32. Overflow (|x| >= 65520) gives +-inf, NaN
 * gives the quiet NaN `sign | 0x7fc00000` (what `f16_to_f32(f16_bits(NaN))` returns), -0 stays -0.
 *   normal halves (|x| >= 2^-14): round the f32 pattern to 10 mantissa bits (carry into the exponent is right);
 *   subnormal halves: the grid is 2^-24, so round(|x| * 2^24) * 2^-24 (both scalings exact).
 */
export const nrRoundHalf = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const sign = bits.bitAnd(u(0x80000000));
  const magnitude = bits.bitAnd(u(0x7fffffff));
  const rounded = magnitude
    .add(u(0xfff))
    .add(magnitude.shiftRight(u(13)).bitAnd(u(1)))
    .bitAnd(u(0xffffe000));
  const normal = pick(rounded.greaterThan(u(0x477fe000)), u(0x7f800000), rounded);
  const subnormal = floatBitsToUint(round(uintBitsToFloat(magnitude).mul(f(2 ** 24))).mul(f(2 ** -24)));
  const finite = pick(magnitude.lessThan(u(0x38800000)), subnormal, normal);
  const result = pick(magnitude.greaterThan(u(0x7f800000)), u(0x7fc00000), finite);
  return uintBitsToFloat(sign.bitOr(result));
}).setLayout({ name: 'nr_round_half', type: 'float', inputs: [{ name: 'value', type: 'float' }] });

/**
 * `publish_fp8(x)` = `decode_e4m3(encode_e4m3(f16_bits(x)))`: x rounded to half, then to E4M3 (ties to even),
 * saturated at +-448 (infinities included), NaN -> +0, the sign of a zero kept.
 *   E4M3 normals (|h| >= 2^-6): round the f32 pattern of the half to 3 mantissa bits;
 *   E4M3 subnormals: the grid is 2^-9, so round(|h| * 512) / 512.
 */
export const nrPublishE4Value = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(nrRoundHalf(value));
  const sign = bits.bitAnd(u(0x80000000));
  const magnitude = bits.bitAnd(u(0x7fffffff));
  const normal = magnitude
    .add(u(0x7ffff))
    .add(magnitude.shiftRight(u(20)).bitAnd(u(1)))
    .bitAnd(u(0xfff00000));
  const subnormal = floatBitsToUint(round(uintBitsToFloat(magnitude).mul(f(512))).mul(f(1 / 512)));
  const finite = min(pick(magnitude.lessThan(u(0x3c800000)), subnormal, normal), u(0x43e00000));
  return uintBitsToFloat(pick(magnitude.greaterThan(u(0x7f800000)), u(0), sign.bitOr(finite)));
}).setLayout({ name: 'nr_publish_e4_value', type: 'float', inputs: [{ name: 'value', type: 'float' }] });

/** WGSL `f16 + f16`: the f32 sum (exact for two halves) rounded to half. */
export const halfAdd = (a: TSLNode, b: TSLNode): TSLNode => nrRoundHalf(a.add(b));

/** One step of a left fold of half adds: `first ? value : halfAdd(sum, value)`. */
export const halfFold = (first: TSLNode, sum: TSLNode, value: TSLNode): TSLNode =>
  pick(first, value, halfAdd(sum, value));

/** `encode_e4m3(f16_bits(x))`: the published E4M3 byte of an f32 (keeps -0 as 0x80, NaN -> 0x00). */
export const e4Code = (value: TSLNode): TSLNode => nrEncodeE4m3(nrF16Bits(value));
