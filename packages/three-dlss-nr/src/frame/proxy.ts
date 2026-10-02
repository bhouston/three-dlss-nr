// The frame's per-pixel math in TSL: the display proxy, the centring, the noise lanes, the conditioning lanes and the
// history's truncation to the half grid.
//
// Port of the helpers of OpenDLSS-NR ports/browser-webgpu/shaders/frame.wgsl and shaders/preprocess.wgsl (MIT,
// (c) 2026 maan, https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/frame.wgsl) to
// three.js TSL; see the reference's docs/frame.md for what they compute and why.
//
// Unlike the network kernels these are not bit-gated: the proxy uses `exp` and `pow`, the noise lanes `log2`, `cos`
// and `sin`, whose last bits are the hardware's. They follow the reference's expression order literally, so on one
// device they agree with the reference's WGSL to the last bit or nearly (frame.gpu.test.ts measures it). Constants
// that are not f32-exact in the WGSL source are written as `Math.fround` of the same decimal, which is what WGSL
// parses them to.

import {
  Fn,
  clamp,
  cos,
  exp,
  float,
  floatBitsToUint,
  int,
  log2,
  max,
  pow,
  sin,
  sqrt,
  uint,
  vec3,
  wgslFn,
} from 'three/tsl';

import { nrRoundF16 } from '../tsl/numerics.js';
import { f, fBits, i, pick, u, type TSLNode } from '../tsl/packed.js';

/** An f32 literal of a decimal constant as WGSL parses it (nearest f32). */
export const fr = (value: number): TSLNode => f(Math.fround(value));

/** WGSL `fma` (TSL has none). Used only where the reference spells `fma`, in the colour grade and intensity. */
export const nrFma = wgslFn('fn nr_fma(a: f32, b: f32, c: f32) -> f32 { return fma(a, b, c); }');

/** `srgb_encode`: linear -> sRGB code value, clamped to [0, 1]. */
export const nrSrgbEncode = Fn(([value]: [TSLNode]) => {
  const bounded = clamp(value, f(0), f(1)).toVar();
  const curve = fr(1.055)
    .mul(pow(bounded, fr(1 / 2.4)))
    .sub(fr(0.055));
  return pick(bounded.lessThanEqual(fr(0.0031308)), fr(12.92).mul(bounded), curve);
}).setLayout({ name: 'nr_srgb_encode', type: 'float', inputs: [{ name: 'value', type: 'float' }] });

/** `srgb_decode`: sRGB code value -> linear, clamped to [0, 1]. */
export const nrSrgbDecode = Fn(([value]: [TSLNode]) => {
  const bounded = clamp(value, f(0), f(1)).toVar();
  const curve = pow(bounded.add(fr(0.055)).div(fr(1.055)), fr(2.4));
  return pick(bounded.lessThanEqual(fr(0.04045)), bounded.div(fr(12.92)), curve);
}).setLayout({ name: 'nr_srgb_decode', type: 'float', inputs: [{ name: 'value', type: 'float' }] });

/**
 * `proxy_component`: the display proxy of one linear scene component - paper-white relative, a soft shoulder above
 * 0.75, sRGB encoded, on the half grid. Non-finite and negative values count as 0 (the NaN test is done on the bits,
 * where the reference's `value == value` could be folded by a compiler).
 */
export const nrProxyComponent = Fn(([value, paperWhite]: [TSLNode, TSLNode]) => {
  const finiteBits = floatBitsToUint(value).bitAnd(u(0x7fffffff)).lessThanEqual(u(0x477fe000)); // |v| <= 65504
  const finite = pick(finiteBits, max(value, f(0)), f(0));
  const relative = finite.div(max(paperWhite, fr(0.05))).toVar();
  const shoulder = fr(0.75).add(fr(0.25).mul(f(1).sub(exp(fr(-5.77078).mul(relative.sub(fr(0.75)))))));
  return nrRoundF16(nrSrgbEncode(pick(relative.greaterThan(fr(0.75)), shoulder, relative)));
}).setLayout({
  name: 'nr_proxy_component',
  type: 'float',
  inputs: [
    { name: 'value', type: 'float' },
    { name: 'paperWhite', type: 'float' },
  ],
});

/** `centre`: what the network is given, the code value centred and scaled down by eight, each step on the half grid. */
export const nrCentre = Fn(([code]: [TSLNode]) =>
  nrRoundF16(nrRoundF16(nrRoundF16(code).sub(f(0.5))).mul(f(0.125))),
).setLayout({ name: 'nr_centre', type: 'float', inputs: [{ name: 'code', type: 'float' }] });

/** `centre` of a vec3. */
export const nrCentre3 = (code: TSLNode): TSLNode => vec3(nrCentre(code.x), nrCentre(code.y), nrCentre(code.z));

/** `hash_uniform`: a u32 hash to a uniform in (0, 1]. */
export const nrHashUniform = Fn(([value]: [TSLNode]) => {
  const first = value.shiftRight(value.shiftRight(u(28)).add(u(4))).bitXor(value);
  const mixed = first.mul(u(0x108ef2d9)).toVar();
  const integer = mixed
    .shiftRight(u(30))
    .bitXor(mixed.shiftRight(u(8)))
    .add(u(1));
  return float(integer).mul(fBits(0x33800000));
}).setLayout({ name: 'nr_hash_uniform', type: 'float', inputs: [{ name: 'value', type: 'uint' }] });

/**
 * `gaussian3`: three Box-Muller Gaussians per padded pixel, on the half grid. Hardware transcendentals: the last bits
 * are the device's (docs/network.md), so these lanes are compared with a tolerance across devices.
 */
export const nrGaussian3 = Fn(([x, y, seed]: [TSLNode, TSLNode, TSLNode]) => {
  const start = x
    .mul(u(0x8da6b343))
    .bitXor(seed.mul(u(0x9e3779b9)))
    .bitXor(y.mul(u(0xd8163841)))
    .bitXor(u(0x243f6a88))
    .toVar();
  const shifted = start
    .shiftRight(start.shiftRight(u(28)).add(u(4)))
    .bitXor(start)
    .mul(u(0x108ef2d9))
    .toVar();
  const base = shifted.shiftRight(u(22)).bitXor(shifted).toVar();
  const u0 = nrHashUniform(base.mul(u(0x2c9277b5)).add(u(0xac564b05)));
  const u1 = nrHashUniform(base.mul(u(0xfa6dc5f9)).add(u(0x4712a88e)));
  const u2 = nrHashUniform(base.mul(u(0xcaa5b80d)).add(u(0x21dd796b)));
  const u3 = nrHashUniform(base.mul(u(0x83232c31)).add(u(0x3463e0ac)));
  const radius0 = sqrt(log2(u0).mul(fBits(0x3f317218)).mul(f(-2))).toVar();
  const radius1 = sqrt(log2(u2).mul(fBits(0x3f317218)).mul(f(-2))).toVar();
  const angle0 = u1.mul(fBits(0x40c90fdb)).toVar();
  const angle1 = u3.mul(fBits(0x40c90fdb));
  return vec3(
    nrRoundF16(radius0.mul(cos(angle0))),
    nrRoundF16(radius0.mul(sin(angle0))),
    nrRoundF16(radius1.mul(cos(angle1))),
  );
}).setLayout({
  name: 'nr_gaussian3',
  type: 'vec3',
  inputs: [
    { name: 'x', type: 'uint' },
    { name: 'y', type: 'uint' },
    { name: 'seed', type: 'uint' },
  ],
});

/**
 * `truncate_half`: truncation toward zero to the half grid (the history is stored truncated, not rounded, so it
 * cannot drift upward frame after frame). Returns the half bit pattern.
 */
export const nrTruncateHalf = Fn(([value]: [TSLNode]) => {
  const bits = floatBitsToUint(value);
  const sign = bits.shiftRight(u(16)).bitAnd(u(0x8000));
  const exponent = bits.shiftRight(u(23)).bitAnd(u(0xff));
  const mantissa = bits.bitAnd(u(0x7fffff));
  const halfExponent = int(exponent).sub(i(112));
  const special = sign.bitOr(pick(mantissa.notEqual(u(0)), u(0x7e00), u(0x7c00)));
  // u32(14 - e) is at most 24 on the path that uses it; clamp so the discarded paths shift by a valid amount.
  const shift = uint(clamp(i(14).sub(halfExponent), i(0), i(31)));
  const subnormal = pick(
    halfExponent.lessThan(i(-10)),
    sign,
    sign.bitOr(mantissa.bitOr(u(0x800000)).shiftRight(shift)),
  );
  const normal = sign.bitOr(uint(max(halfExponent, i(0))).shiftLeft(u(10))).bitOr(mantissa.shiftRight(u(13)));
  return pick(
    exponent.equal(u(0xff)),
    special,
    pick(
      halfExponent.greaterThanEqual(i(31)),
      sign.bitOr(u(0x7c00)),
      pick(halfExponent.lessThanEqual(i(0)), subnormal, normal),
    ),
  );
}).setLayout({ name: 'nr_truncate_half', type: 'uint', inputs: [{ name: 'value', type: 'float' }] });

/** Conditioning values the lanes 10-14 are built from (`frame.wgsl` / `preprocess.wgsl` params). */
export interface ConditioningNodes {
  style: TSLNode;
  localTone: TSLNode;
  localStructure: TSLNode;
  skinStructure: TSLNode;
  autoMask: TSLNode;
}

/** Feature lanes 10-14 (lane 15 is 0): style / 128, then the four conditioning lanes on the half grid. */
export function conditioningLanes({
  style,
  localTone,
  localStructure,
  skinStructure,
  autoMask,
}: ConditioningNodes): TSLNode[] {
  const masked = autoMask.greaterThan(f(0));
  return [
    style.div(f(128)),
    nrRoundF16(localTone),
    nrRoundF16(pick(masked, f(1), localStructure)),
    nrRoundF16(pick(masked, pick(skinStructure.lessThan(f(0)), localStructure, skinStructure), f(-1))),
    nrRoundF16(pick(masked, localStructure, f(-1))),
  ];
}

/**
 * The source pixel of a padded coordinate: inside the valid image itself, outside it mirrored (`2 * valid - x - 2`).
 * The reference reads out of bounds where one mirror is not enough (a valid size under half the field); here the
 * mirrored coordinate is clamped into the image instead. Identical wherever the reference is in bounds.
 */
export function mirroredSource(x: TSLNode, valid: number): TSLNode {
  const mirrored = clamp(i(2 * valid - 2).sub(int(x)), i(0), i(valid - 1));
  return pick(x.lessThan(u(valid)), x, uint(mirrored));
}
