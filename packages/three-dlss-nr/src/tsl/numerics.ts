// Port of OpenDLSS-NR ports/browser-webgpu/shaders/numerics.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/numerics.wgsl) to three.js TSL,
// plus `publish_e4_code` of src/matmul/packed-activation.js (`nrPublishE4CodeGemm`).
//
// The publication grid in TSL. Every helper is a layout `Fn` (a real WGSL `fn`, so kernels that call it stay small)
// with the same name, arguments and arithmetic as its WGSL twin. Most helpers are branch-free; half publication
// uses a short exact path for normal inputs and the original fallback for exceptional values. Every helper is
// checked exhaustively on the GPU against the reference's numerics fixture and the TS oracle (numerics.gpu.test.ts).
//
// Rules this file keeps (design section 3):
//   * No f16 type: everything is f32 or integer arithmetic on bit patterns.
//   * No exp2 / log2 / pow / fma / mix / inverseSqrt: powers of two come from the exponent field (`nrPow2`), and
//     every `a * b + c` has an exact product (R4).
//   * Every literal is typed (`u`, `i`, `f`; R1-R3), and integer comparisons compare integers (R2).
//   * `pick(cond, ifTrue, ifFalse)` is TSL's argument order - the reverse of WGSL's `pick(f, t, cond)`.
//   * NaN tests are done on bit patterns rather than `x != x`, which a compiler may fold.

import {
  Fn,
  If,
  abs,
  clamp,
  countLeadingZeros,
  float,
  floatBitsToUint,
  int,
  max,
  min,
  round,
  trunc,
  uint,
  uintBitsToFloat,
} from 'three/tsl';

import { f, i, pick, u, type TSLNode } from './packed.js';

/** 2^e (`e`: i32), exactly, built from the exponent field. Callers stay inside the normal range (|e| < 127). */
export const nrPow2 = Fn(([e]: [TSLNode]) =>
  uintBitsToFloat(uint(clamp(e, i(-126), i(127)).add(i(127))).shiftLeft(u(23))),
).setLayout({ name: 'nr_pow2', type: 'float', inputs: [{ name: 'e', type: 'int' }] });

/** Shift right (`shift`: u32) with round-to-nearest-even; 0 for shifts above 31. */
export const nrRoundShiftRightEven = Fn(([value, shift]: [TSLNode, TSLNode]) => {
  // WGSL takes runtime shift amounts modulo 32, so the out-of-range paths are computed on a clamped shift and
  // discarded by the selects below.
  const s = min(max(shift, u(1)), u(31));
  const quotient = value.shiftRight(s);
  const remainder = value.bitAnd(u(1).shiftLeft(s).sub(u(1)));
  const halfway = u(1).shiftLeft(s.sub(u(1)));
  const up = remainder.greaterThan(halfway).or(remainder.equal(halfway).and(quotient.bitAnd(u(1)).notEqual(u(0))));
  const rounded = quotient.add(pick(up, u(1), u(0)));
  return pick(shift.equal(u(0)), value, pick(shift.greaterThan(u(31)), u(0), rounded));
}).setLayout({
  name: 'nr_round_shift_right_even',
  type: 'uint',
  inputs: [
    { name: 'value', type: 'uint' },
    { name: 'shift', type: 'uint' },
  ],
});

/** IEEE binary16 bit pattern of an f32, round-to-nearest-even (`f16_bits`). */
const nrF16BitsFallback = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const sign = bits.shiftRight(u(16)).bitAnd(u(0x8000));
  const exponent = bits.shiftRight(u(23)).bitAnd(u(0xff));
  const mantissa = bits.bitAnd(u(0x7fffff));
  const halfExponent = int(exponent).sub(i(112));
  // exponent == 0xff: infinity or a quiet NaN.
  const special = sign.bitOr(pick(mantissa.notEqual(u(0)), u(0x7e00), u(0x7c00)));
  // halfExponent <= 0: subnormal half (or zero below 2^-25). u32(14 - e) of a very negative e is > 31 -> 0.
  const subnormal = pick(
    halfExponent.lessThan(i(-10)),
    sign,
    sign.bitOr(nrRoundShiftRightEven(mantissa.bitOr(u(0x800000)), uint(i(14).sub(halfExponent)))),
  );
  // Normal: round the mantissa, carry into the exponent, overflow to infinity.
  const rounded = nrRoundShiftRightEven(mantissa, u(13));
  const carry = rounded.equal(u(0x400));
  const normalExponent = halfExponent.add(pick(carry, i(1), i(0)));
  const normalMantissa = pick(carry, u(0), rounded);
  const normal = pick(
    normalExponent.greaterThanEqual(i(31)),
    sign.bitOr(u(0x7c00)),
    sign.bitOr(uint(normalExponent).shiftLeft(u(10))).bitOr(normalMantissa),
  );
  return pick(
    exponent.equal(u(0xff)),
    special,
    pick(
      halfExponent.greaterThanEqual(i(31)),
      sign.bitOr(u(0x7c00)),
      pick(halfExponent.lessThanEqual(i(0)), subnormal, normal),
    ),
  );
}).setLayout({ name: 'nr_f16_bits_fallback', type: 'uint', inputs: [{ name: 'value', type: 'float' }] });

/** Exact half publication with a small common path for normal finite inputs. */
export const nrF16Bits = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const exponent = bits.shiftRight(u(23)).bitAnd(u(0xff));
  const result = u(0).toVar();
  // Carry into exponent 31 correctly produces infinity. Subnormals and NaNs
  // retain the fully specified fallback, without executing it on normal values.
  If(exponent.greaterThanEqual(u(113)).and(exponent.lessThanEqual(u(142))), () => {
    const sign = bits.shiftRight(u(16)).bitAnd(u(0x8000));
    const magnitude = bits.bitAnd(u(0x7fffffff));
    const rounded = magnitude.add(u(0xfff)).add(magnitude.shiftRight(u(13)).bitAnd(u(1)));
    result.assign(sign.bitOr(rounded.shiftRight(u(13)).sub(u(112 * 1024))));
  }).Else(() => {
    result.assign(nrF16BitsFallback(value));
  });
  return result;
}).setLayout({ name: 'nr_f16_bits', type: 'uint', inputs: [{ name: 'value', type: 'float' }] });

/**
 * f32 value of an IEEE binary16 bit pattern, assembled on the bit pattern (`f16_to_f32`): infinities and NaN
 * payloads survive; subnormal halves are normalized with countLeadingZeros.
 */
export const nrF16ToF32 = Fn(([bits]: [TSLNode]) => {
  const sign = bits.bitAnd(u(0x8000)).shiftLeft(u(16));
  const exponent = bits.shiftRight(u(10)).bitAnd(u(0x1f));
  const mantissa = bits.bitAnd(u(0x3ff));
  const shift = countLeadingZeros(mantissa).sub(u(21)); // brings the leading one up to bit 10
  const normalized = mantissa.shiftLeft(shift).bitAnd(u(0x3ff));
  const subnormal = sign.bitOr(u(113).sub(shift).shiftLeft(u(23))).bitOr(normalized.shiftLeft(u(13)));
  const zeroOrSubnormal = pick(mantissa.equal(u(0)), sign, subnormal);
  const special = sign.bitOr(u(0x7f800000)).bitOr(mantissa.shiftLeft(u(13)));
  const normal = sign.bitOr(exponent.add(u(112)).shiftLeft(u(23))).bitOr(mantissa.shiftLeft(u(13)));
  return uintBitsToFloat(pick(exponent.equal(u(0)), zeroOrSubnormal, pick(exponent.equal(u(0x1f)), special, normal)));
}).setLayout({ name: 'nr_f16_to_f32', type: 'float', inputs: [{ name: 'bits', type: 'uint' }] });

/** One publication to the half grid (`round_f16`). */
export const nrRoundF16 = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const magnitude = bits.bitAnd(u(0x7fffffff));
  const result = f(0).toVar();
  // A normal, finite half can be rounded directly on the f32 bit pattern.
  // 65520 is the halfway point that rounds to infinity; leave it and every
  // subnormal/special value to the original conversion (including signed zero).
  If(magnitude.greaterThanEqual(u(0x38800000)).and(magnitude.lessThan(u(0x477ff000))), () => {
    result.assign(
      uintBitsToFloat(
        bits
          .add(u(0xfff))
          .add(bits.shiftRight(u(13)).bitAnd(u(1)))
          .bitAnd(u(0xffffe000)),
      ),
    );
  }).Else(() => {
    result.assign(nrF16ToF32(nrF16Bits(value)));
  });
  return result;
}).setLayout({
  name: 'nr_round_f16',
  type: 'float',
  inputs: [{ name: 'value', type: 'float' }],
});

/** E4M3FN value of a byte (`decode_e4m3`); the NaN code reads as a signed zero, 0x80 as -0. */
export const nrDecodeE4m3 = Fn(([bits]: [TSLNode]) => {
  const negative = bits.bitAnd(u(0x80)).notEqual(u(0));
  const exponent = bits.shiftRight(u(3)).bitAnd(u(0x0f));
  const mantissa = bits.bitAnd(u(0x07));
  const subnormal = float(mantissa).mul(nrPow2(i(-9)));
  const normal = f(1)
    .add(float(mantissa).mul(f(0.125)))
    .mul(nrPow2(int(exponent).sub(i(7))));
  const value = pick(
    exponent.equal(u(0)),
    subnormal,
    pick(exponent.equal(u(15)).and(mantissa.equal(u(7))), f(0), normal),
  );
  return pick(negative, value.negate(), value);
}).setLayout({ name: 'nr_decode_e4m3', type: 'float', inputs: [{ name: 'bits', type: 'uint' }] });

/**
 * E4M3FN code of an already half-rounded value given as half bits (`encode_e4m3`): RNE, finite saturation at 448,
 * NaN -> +0, and the sign of a zero survives (-0 -> 0x80).
 */
export const nrEncodeE4m3 = Fn(([halfBits]: [TSLNode]) => {
  const isNaN = halfBits
    .bitAnd(u(0x7c00))
    .equal(u(0x7c00))
    .and(halfBits.bitAnd(u(0x03ff)).notEqual(u(0)));
  const isZero = halfBits.bitAnd(u(0x7fff)).equal(u(0));
  const negative = pick(halfBits.bitAnd(u(0x8000)).notEqual(u(0)), u(0x80), u(0));
  const exponent = halfBits.shiftRight(u(10)).bitAnd(u(0x1f));
  const mantissa = halfBits.bitAnd(u(0x3ff));
  // Below 2^-6 the E4M3 grid is subnormal: the half significand is shifted down onto a fixed step of 2^-9.
  // (16 - exponent is only used for exponent <= 8, so it never wraps.)
  const significand = pick(exponent.equal(u(0)), mantissa, u(1024).add(mantissa));
  const shift = pick(exponent.equal(u(0)), u(15), u(16).sub(min(exponent, u(16))));
  const subnormalCode = min(nrRoundShiftRightEven(significand, shift), u(8));
  const e4Rounded = nrRoundShiftRightEven(mantissa, u(7));
  const e4Carry = e4Rounded.equal(u(8));
  const e4Exponent = exponent.sub(u(8)).add(pick(e4Carry, u(1), u(0)));
  const e4Mantissa = pick(e4Carry, u(0), e4Rounded);
  const overflow = e4Exponent.greaterThan(u(15)).or(e4Exponent.equal(u(15)).and(e4Mantissa.greaterThan(u(6))));
  const normalCode = pick(overflow, u(0x7e), e4Exponent.shiftLeft(u(3)).bitOr(e4Mantissa));
  const code = pick(exponent.equal(u(31)), u(0x7e), pick(exponent.lessThanEqual(u(8)), subnormalCode, normalCode));
  return pick(isNaN, u(0), pick(isZero, halfBits.shiftRight(u(8)).bitAnd(u(0x80)), negative.bitOr(code)));
}).setLayout({ name: 'nr_encode_e4m3', type: 'uint', inputs: [{ name: 'half_bits', type: 'uint' }] });

/**
 * The FP8 GEMM's E4 publication of an f32 (`publish_e4_code`, src/matmul/packed-activation.js): NaN -> 0, saturate at
 * 448, E4 subnormals `round(m * 512)`, normals RNE on the f32 pattern; **-0 -> 0x00** (sign only for `value < 0`).
 */
export const nrPublishE4CodeGemm = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const isNaN = bits.bitAnd(u(0x7fffffff)).greaterThan(u(0x7f800000));
  const magnitude = min(abs(value), f(448));
  const magnitudeBits = floatBitsToUint(magnitude);
  const subnormalCode = uint(round(magnitude.mul(f(512))));
  const rounded = magnitudeBits
    .add(u(0x7ffff))
    .add(magnitudeBits.shiftRight(u(20)).bitAnd(u(1)))
    .bitAnd(u(0xfff00000));
  const normalCode = rounded.shiftRight(u(20)).sub(u(960));
  const code = pick(magnitude.lessThan(f(0.015625)), subnormalCode, normalCode).bitOr(
    pick(value.lessThan(f(0)), u(128), u(0)),
  );
  return pick(isNaN.or(magnitude.equal(f(0))), u(0), code);
}).setLayout({ name: 'nr_publish_e4_code_gemm', type: 'uint', inputs: [{ name: 'value', type: 'float' }] });

/**
 * MpCubicSiLU (`mp_cubic_silu`): five half publications; the two inner multiply-adds have exact f32 products (a half
 * times an 8-bit constant, then two halves), so contraction cannot change them.
 */
export const nrMpCubicSilu = Fn(([value]: [TSLNode]) => {
  const bounded = nrRoundF16(clamp(value, f(-4), f(4)));
  const absolute = nrRoundF16(abs(bounded));
  const inner = nrRoundF16(f(-0.055908203125).mul(absolute).add(f(0.447265625)));
  const polynomial = nrRoundF16(bounded.mul(inner).add(f(0.89453125)));
  return nrRoundF16(value.mul(polynomial));
}).setLayout({ name: 'nr_mp_cubic_silu', type: 'float', inputs: [{ name: 'value', type: 'float' }] });

/**
 * The window blocks' attention weight (`exp_weight`): an affine map on the score (exact product, one rounding),
 * clamped, dropped into the half exponent field.
 */
export const nrExpWeight = Fn(([score]: [TSLNode]) => {
  const affine = clamp(nrRoundF16(score.mul(f(0.044921875)).add(f(1.30078125))), f(1.03125), f(1.5693359375));
  return nrF16ToF32(nrF16Bits(affine).shiftLeft(u(5)).add(u(0x8000)).bitAnd(u(0xffff)));
}).setLayout({ name: 'nr_exp_weight', type: 'float', inputs: [{ name: 'score', type: 'float' }] });

/**
 * The ViT's variant (`vit_exp_weight`): 4-bit shift, the half constants native holds; the f32 evaluation of the half
 * fused multiply-add is exact here (see numerics.wgsl).
 */
export const nrVitExpWeight = Fn(([score]: [TSLNode]) => {
  const affine = clamp(nrRoundF16(score.mul(f(0.08953857421875)).add(f(1.708984375))), f(1.439453125), f(1.9775390625));
  return nrF16ToF32(nrF16Bits(affine).shiftLeft(u(4)).add(u(0x4000)).bitAnd(u(0xffff)));
}).setLayout({ name: 'nr_vit_exp_weight', type: 'float', inputs: [{ name: 'score', type: 'float' }] });

/**
 * Unbiased exponent field of |value| (`normal_exponent`), as i32. Read straight off the bit pattern: the reference's
 * `bitcast<u32>(abs(value))` is the same field (the sign is masked off by `& 0xff`), and FXC (D3D12 without DXC)
 * fails to compile `asuint(abs(x))` in this position with E_FAIL.
 */
export const nrNormalExponent = Fn(([value]: [TSLNode]) =>
  int(floatBitsToUint(value).shiftRight(u(23)).bitAnd(u(0xff))).sub(i(127)),
).setLayout({ name: 'nr_normal_exponent', type: 'int', inputs: [{ name: 'value', type: 'float' }] });

/** `max(normal_exponent, -6)`: the exponent an E4M3 value aligns at. */
export const nrE4m3Exponent = Fn(([value]: [TSLNode]) => max(nrNormalExponent(value), i(-6))).setLayout({
  name: 'nr_e4m3_exponent',
  type: 'int',
  inputs: [{ name: 'value', type: 'float' }],
});

/** `max(normal_exponent, -14)`: the exponent a half value aligns at. */
export const nrF16Exponent = Fn(([value]: [TSLNode]) => max(nrNormalExponent(value), i(-14))).setLayout({
  name: 'nr_f16_exponent',
  type: 'int',
  inputs: [{ name: 'value', type: 'float' }],
});

/** The shared exponent of a step that starts from this accumulator (`f13_start`); -21 is the empty value. */
export const nrF13Start = Fn(([accumulator]: [TSLNode]) =>
  pick(accumulator.notEqual(f(0)), nrF16Exponent(accumulator), i(-21)),
).setLayout({ name: 'nr_f13_start', type: 'int', inputs: [{ name: 'accumulator', type: 'float' }] });

/** Widen the shared exponent to cover one E4M3 product (`f13_cover`); a zero operand is skipped, not clamped. */
export const nrF13Cover = Fn(([maximumExponent, a, b]: [TSLNode, TSLNode, TSLNode]) =>
  pick(
    a.equal(f(0)).or(b.equal(f(0))),
    maximumExponent,
    max(maximumExponent, nrE4m3Exponent(a).add(nrE4m3Exponent(b))),
  ),
).setLayout({
  name: 'nr_f13_cover',
  type: 'int',
  inputs: [
    { name: 'maximum_exponent', type: 'int' },
    { name: 'a', type: 'float' },
    { name: 'b', type: 'float' },
  ],
});

/** One term of the aligned sum, in units of 2^(max - 13) (`f13_term`); `scale` is `nrPow2(13 - max)`. */
export const nrF13Term = Fn(([value, scale]: [TSLNode, TSLNode]) => int(trunc(value.mul(scale)))).setLayout({
  name: 'nr_f13_term',
  type: 'int',
  inputs: [
    { name: 'value', type: 'float' },
    { name: 'scale', type: 'float' },
  ],
});

/** The aligned sum, published once to the half grid (`f13_finish`). */
export const nrF13Finish = Fn(([units, maximumExponent]: [TSLNode, TSLNode]) =>
  nrRoundF16(float(units).mul(nrPow2(maximumExponent.sub(i(13))))),
).setLayout({
  name: 'nr_f13_finish',
  type: 'float',
  inputs: [
    { name: 'units', type: 'int' },
    { name: 'maximum_exponent', type: 'int' },
  ],
});

/**
 * Exact signed integer times 2^binaryExponent -> half value (`fixed_to_f16`), for the f16 step whose fixed-point
 * total carries more significand than a half holds.
 */
export const nrFixedToF16 = Fn(([fixedSum, binaryExponent]: [TSLNode, TSLNode]) => {
  const negative = fixedSum.lessThan(i(0));
  const magnitude = uint(abs(fixedSum));
  const msb = u(31).sub(countLeadingZeros(magnitude));
  const valueExponent = int(msb).add(binaryExponent);
  const sign = pick(negative, u(0x8000), u(0));
  // Normal half: round the magnitude to 11 significant bits. (The unused shift of each select arm may wrap; WGSL
  // takes it modulo 32 and the result is discarded.)
  const significand = pick(
    msb.greaterThan(u(10)),
    nrRoundShiftRightEven(magnitude, msb.sub(u(10))),
    magnitude.shiftLeft(u(10).sub(msb)),
  );
  const carry = significand.greaterThanEqual(u(2048));
  const roundedExponent = valueExponent.add(pick(carry, i(1), i(0)));
  const roundedSignificand = pick(carry, u(1024), significand);
  const normal = pick(
    roundedExponent.greaterThanEqual(i(16)),
    sign.bitOr(u(0x7c00)),
    sign.bitOr(uint(roundedExponent.add(i(15))).shiftLeft(u(10))).bitOr(roundedSignificand.sub(u(1024))),
  );
  // Subnormal half.
  const subnormalScale = binaryExponent.add(i(24));
  const mantissa = pick(
    subnormalScale.greaterThanEqual(i(0)),
    magnitude.shiftLeft(uint(subnormalScale)),
    nrRoundShiftRightEven(magnitude, uint(subnormalScale.negate())),
  );
  const subnormal = sign.bitOr(min(mantissa, u(1024)));
  const halfBits = pick(valueExponent.greaterThanEqual(i(-14)), normal, subnormal);
  return pick(fixedSum.equal(i(0)), f(0), nrF16ToF32(halfBits));
}).setLayout({
  name: 'nr_fixed_to_f16',
  type: 'float',
  inputs: [
    { name: 'fixed_sum', type: 'int' },
    { name: 'binary_exponent', type: 'int' },
  ],
});

/**
 * One k16 group of an FP8 step (`ada_fp8_fdpa16`) over operands the caller already holds: `a`, `b` are 16 f32 nodes
 * holding E4M3 values, `accumulator` an f32 node holding a half. Emitted inline (not a WGSL `fn`), as the reference's
 * GEMM inlines the same f13 calls over its registers. An infinite or NaN accumulator passes through untouched.
 *
 * Note: the composed reference GEMM does NOT pass non-finite accumulators through (design Appendix A.1); kernels
 * port their own composed form from the f13 helpers. This is the numerics.wgsl form, tested against FDP8.
 */
export function nrFdpa16(a: readonly TSLNode[], b: readonly TSLNode[], accumulator: TSLNode): TSLNode {
  if (a.length !== b.length || a.length > 16) throw new RangeError('nrFdpa16 takes up to 16 operand pairs');
  let maximumExponent = nrF13Start(accumulator);
  for (let k = 0; k < a.length; ++k) maximumExponent = nrF13Cover(maximumExponent, a[k], b[k]);
  const maxE = maximumExponent.toVar();
  const scale = nrPow2(i(13).sub(maxE)).toVar();
  let units = nrF13Term(accumulator, scale);
  for (let k = 0; k < a.length; ++k) units = units.add(nrF13Term(a[k].mul(b[k]), scale));
  const finite = floatBitsToUint(accumulator).bitAnd(u(0x7f800000)).notEqual(u(0x7f800000));
  return pick(finite, nrF13Finish(units, maxE), accumulator);
}

/**
 * The f16 step (`ada_f16_fdpa8`): up to 8 products of halves (f32 nodes) against 24 fractional bits, accumulated as an
 * exact i32 and converted with `nrFixedToF16`. Emitted inline.
 */
export function nrFdpaF16x8(a: readonly TSLNode[], b: readonly TSLNode[], accumulator: TSLNode): TSLNode {
  if (a.length !== b.length || a.length > 8) throw new RangeError('nrFdpaF16x8 takes up to 8 operand pairs');
  let maximumExponent = nrF13Start(accumulator);
  for (let k = 0; k < a.length; ++k) {
    maximumExponent = pick(
      a[k].notEqual(f(0)).and(b[k].notEqual(f(0))),
      max(maximumExponent, nrF16Exponent(a[k]).add(nrF16Exponent(b[k]))),
      maximumExponent,
    );
  }
  const maxE = maximumExponent.toVar();
  const scale = nrPow2(i(24).sub(maxE)).toVar();
  let units = int(trunc(accumulator.mul(scale)));
  for (let k = 0; k < a.length; ++k) units = units.add(int(trunc(a[k].mul(b[k]).mul(scale))));
  return nrFixedToF16(units, maxE.sub(i(24)));
}
