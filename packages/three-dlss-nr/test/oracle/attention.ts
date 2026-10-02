// CPU oracles of the reference's two attentions, for tests.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT, pinned at 9d08f41). Semantics of
//   * the composed production window attention of ports/browser-webgpu/src/window/ (`windowAttentionCode()`, entry
//     `attend_window_tiled`, design Appendix A.3), and
//   * ports/browser-webgpu/shaders/vit.wgsl (`vit_normalize`, `vit_attend`, A.4),
// written over the TS oracle in src/numerics/oracle.ts. Written from the WGSL, not from our TSL, and literal on
// purpose: every f16 operation of the reference (`f16 * f16`, `f16 + f16`, `f16(f32)`) is an f32-or-wider operation
// followed by `roundF16`, every f32 step goes through `Math.fround`, the packed exponent bytes and their rotate-max
// are computed as the WGSL computes them, and the reductions keep the reference's order. Slow; meant for small shapes.

import {
  adaFp8Fdpa16,
  e4m3Exponent,
  e4m3FromF16Bits,
  e4m3ToNumber,
  expWeight,
  f16Bits,
  f16Exponent,
  f16ToNumber,
  f32FromBits,
  inverseTiledToken,
  roundF16,
  vitExpWeight,
} from '../../src/numerics/oracle.js';

const fround = Math.fround;

/** `publish_fp8`: the E4M3 publication as a value (`decode_e4m3(encode_e4m3(f16_bits(x)))`; keeps -0, NaN -> +0). */
export const publishFp8 = (value: number): number => e4m3ToNumber(e4m3FromF16Bits(f16Bits(value)));

/** `publish_e4_code` / `encode_e4m3(f16_bits(x))`: the published byte. */
export const publishE4Code = (value: number): number => e4m3FromF16Bits(f16Bits(value));

/** Product of two halves rounded to half, as WGSL `f16 * f16` (the product is exact in f32, so one rounding). */
const halfMul = (a: number, b: number): number => roundF16(a * b);
/** Sum of two halves rounded to half, as WGSL `f16 + f16`. */
const halfAdd = (a: number, b: number): number => roundF16(a + b);
/** `f16(1.0 / sqrt(f32(x)))`. */
const halfRsqrt = (x: number): number => roundF16(fround(1 / fround(Math.sqrt(x))));
/** `f16(1.0 / f32(x))`. */
const halfRecip = (x: number): number => roundF16(fround(1 / x));

/** `bitcast<f32>(u32(e) << 23u)` with u32 wrap-around, as the composed kernel builds its alignments. */
const pow2Bits = (biased: number): number => f32FromBits(((biased >>> 0) << 23) >>> 0);

/** `packed_window_exponents`: byte `e4m3_exponent(v) + 106` per nonzero lane (zero for +-0), lane 0 lowest. */
export function packedWindowExponents(v: ArrayLike<number>): number {
  let word = 0;
  for (let lane = 0; lane < 4; ++lane) {
    const byte = v[lane] !== 0 ? (e4m3Exponent(v[lane]) + 106) >>> 0 : 0;
    word = (word | ((byte & 0xff) << (lane * 8))) >>> 0;
  }
  return word;
}

/** `maximum_window_exponent`: the largest byte of `packed` via the rotate-and-max chain, then max with `current`. */
export function maximumWindowExponent(current: number, packed: number): number {
  const p = packed >>> 0;
  const rotated8 = ((p << 8) | (p >>> 24)) >>> 0;
  const rotated16 = ((p << 16) | (p >>> 16)) >>> 0;
  const rotated24 = ((p << 24) | (p >>> 8)) >>> 0;
  return Math.max(current, Math.max(Math.max(p, rotated8), Math.max(rotated16, rotated24)) >>> 24);
}

/**
 * One 16-product group of the composed window kernel (`tiled_qk_*` / `tiled_value_*`): start the byte maximum at 191
 * (exponent -21) or at the accumulator's half exponent + 212, widen it with four packed byte-sums, align, truncate,
 * sum in f32, map a zero sum to +0, scale back and round to half. `a`/`b` are 16 values, `pa`/`pb` their four packed
 * exponent words, `productScale` multiplies each product before it is aligned (the value groups use 4 * 4 operands
 * and `136 - exponent`).
 */
function windowGroup(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  pa: ArrayLike<number>,
  pb: ArrayLike<number>,
  accumulator: number,
  scaledProducts: boolean,
): number {
  let maximum = 191;
  if (accumulator !== 0) maximum = (f16Exponent(accumulator) + 212) >>> 0;
  for (let c = 0; c < 4; ++c) maximum = maximumWindowExponent(maximum, (pa[c] + pb[c]) >>> 0);
  const exponent = maximum - 212;
  const alignment = pow2Bits(140 - exponent);
  let sum = Math.trunc(fround(accumulator * alignment));
  if (scaledProducts) {
    // half-value: both operands carry a factor 4 (exact), the product is formed in half, aligned by 2^(9 - e).
    const productAlignment = pow2Bits(136 - exponent);
    for (let i = 0; i < 16; ++i) {
      const product = halfMul(halfMul(a[i], 4), halfMul(b[i], 4));
      sum = fround(sum + Math.trunc(fround(product * productAlignment)));
    }
  } else {
    for (let i = 0; i < 16; ++i) sum = fround(sum + Math.trunc(fround(fround(a[i] * b[i]) * alignment)));
  }
  return roundF16(fround((sum === 0 ? 0 : sum) * pow2Bits(exponent + 114)));
}

export interface OracleWindowAttentionArgs {
  width: number;
  height: number;
  heads: number;
  shiftX: number;
  shiftY: number;
  /** Raw qkv half bit patterns `[width * height][heads * 96]` (per head: q 0..31, k 32..63, v 64..95). */
  qkv: Uint16Array;
  /** Prior half bit patterns `[heads][64 natural query][64 natural key]`. */
  prior: Uint16Array;
  /** Per-head f32 scales. */
  scales: Float32Array;
}

/**
 * The composed `attend_window_tiled` with `WINDOW_USE_RELATIVE_BIAS = 1`: E4M3 output bytes `[width * height][heads *
 * 32]` (every token is in exactly one window of a phase, so every byte is written).
 */
export function oracleWindowAttention(args: OracleWindowAttentionArgs): Uint8Array {
  const { width, height, heads, shiftX, shiftY, qkv, prior, scales } = args;
  const channels = heads * 32;
  const stride = channels * 3;
  const tokens = width * height;
  if (qkv.length < tokens * stride) throw new Error('oracleWindowAttention: qkv too short');
  const output = new Uint8Array(tokens * channels);
  const written = new Uint8Array(tokens);
  const windowsX = Math.ceil((width + shiftX) / 8);
  const windowsY = Math.ceil((height + shiftY) / 8);
  const half = (index: number): number => f16ToNumber(qkv[index]);

  for (let head = 0; head < heads; ++head) {
    const scale = roundF16(scales[head]); // f16(scale)
    for (let window = 0; window < windowsX * windowsY; ++window) {
      const wx = (window % windowsX) * 8 - shiftX;
      const wy = Math.floor(window / windowsX) * 8 - shiftY;
      // Natural window token -> field token index, or -1 outside the field.
      const fieldToken = (token: number): number => {
        const x = wx + (token % 8);
        const y = wy + Math.floor(token / 8);
        return x >= 0 && y >= 0 && x < width && y < height ? y * width + x : -1;
      };

      // Cosine norms per natural token: eight lanes, each folding its four squares (nr_norm_fma), then the
      // stride-4/2/1 tree of half adds, then f16(1/sqrt).
      const normQ = new Float64Array(64);
      const normK = new Float64Array(64);
      for (let token = 0; token < 64; ++token) {
        const field = fieldToken(token);
        const lanes = (offset: number): number[] => {
          const values: number[] = [];
          for (let component = 0; component < 8; ++component) {
            const at = (c: number): number => (field < 0 ? 0 : half(field * stride + head * 96 + offset + c));
            const v0 = at(component);
            const v8 = at(component + 8);
            const v16 = at(component + 16);
            const v24 = at(component + 24);
            const low = roundF16(fround(v0 * v0 + halfMul(v16, v16)));
            const high = roundF16(fround(v8 * v8 + halfMul(v24, v24)));
            values.push(halfAdd(low, high));
          }
          return values;
        };
        const tree = (x: number[]): number => {
          for (let stride4 = 4; stride4 > 0; stride4 >>= 1) {
            for (let c = 0; c < stride4; ++c) x[c] = halfAdd(x[c], x[c + stride4]);
          }
          return x[0];
        };
        normQ[token] = halfRsqrt(tree(lanes(0)));
        normK[token] = halfRsqrt(tree(lanes(32)));
      }

      // Staged operands. K by natural key, V by physical key, Q by natural query; all published to E4M3.
      const k = new Float64Array(64 * 32);
      const v = new Float64Array(64 * 32);
      const q = new Float64Array(64 * 32);
      for (let token = 0; token < 64; ++token) {
        const field = fieldToken(token);
        if (field < 0) continue;
        const base = field * stride + head * 96;
        for (let c = 0; c < 32; ++c) {
          k[token * 32 + c] = publishFp8(halfMul(half(base + 32 + c), normK[token]));
          q[token * 32 + c] = publishFp8(halfMul(halfMul(half(base + c), normQ[token]), scale));
        }
      }
      for (let physical = 0; physical < 64; ++physical) {
        const field = fieldToken(inverseTiledToken(physical));
        if (field < 0) continue;
        for (let c = 0; c < 32; ++c) v[physical * 32 + c] = publishFp8(half(field * stride + head * 96 + 64 + c));
      }
      const packQ = (query: number, group: number): number =>
        packedWindowExponents(q.subarray(query * 32 + group * 4, query * 32 + group * 4 + 4));
      const packK = (key: number, group: number): number =>
        packedWindowExponents(k.subarray(key * 32 + group * 4, key * 32 + group * 4 + 4));

      for (let query = 0; query < 64; ++query) {
        const target = fieldToken(query);
        // Scores in physical key order: two 16-channel groups seeded with the prior, then exp_weight.
        const scores = new Float64Array(64);
        for (let physical = 0; physical < 64; ++physical) {
          const key = inverseTiledToken(physical);
          let score = f16ToNumber(prior[(head * 64 + query) * 64 + key]);
          for (let start = 0; start < 32; start += 16) {
            const g = start / 4;
            score = windowGroup(
              q.subarray(query * 32 + start, query * 32 + start + 16),
              k.subarray(key * 32 + start, key * 32 + start + 16),
              [packQ(query, g), packQ(query, g + 1), packQ(query, g + 2), packQ(query, g + 3)],
              [packK(key, g), packK(key, g + 1), packK(key, g + 2), packK(key, g + 3)],
              score,
              false,
            );
          }
          scores[physical] = expWeight(score);
        }
        // tiled_softmax_pair / tiled_softmax_sum over physical indices, every add a half add.
        const pair = (p: number, parity: number): number => {
          const key = p * 2 + parity;
          const blocks01 = halfAdd(scores[key], scores[key + 8]);
          const blocks23 = halfAdd(scores[key + 16], scores[key + 24]);
          const blocks45 = halfAdd(scores[key + 32], scores[key + 40]);
          const blocks67 = halfAdd(scores[key + 48], scores[key + 56]);
          return halfAdd(halfAdd(halfAdd(blocks01, blocks23), blocks45), blocks67);
        };
        const sumParity = (parity: number): number =>
          halfAdd(halfAdd(halfAdd(pair(0, parity), pair(1, parity)), pair(2, parity)), pair(3, parity));
        const reciprocal = halfRecip(halfAdd(sumParity(0), sumParity(1)));
        const weights = new Float64Array(64);
        for (let physical = 0; physical < 64; ++physical) {
          weights[physical] = publishFp8(halfMul(scores[physical], reciprocal));
        }
        if (target < 0) continue;
        // Values: four groups over physical keys 0-15, 16-31, 32-47, 48-63.
        for (let component = 0; component < 32; ++component) {
          let value = 0;
          for (let start = 0; start < 64; start += 16) {
            const vs = new Float64Array(16);
            for (let i = 0; i < 16; ++i) vs[i] = v[(start + i) * 32 + component];
            const pw: number[] = [];
            const pv: number[] = [];
            for (let c = 0; c < 4; ++c) {
              pw.push(packedWindowExponents(weights.subarray(start + c * 4, start + c * 4 + 4)));
              pv.push(packedWindowExponents(vs.subarray(c * 4, c * 4 + 4)));
            }
            value = windowGroup(weights.subarray(start, start + 16), vs, pw, pv, value, true);
          }
          output[target * channels + head * 32 + component] = publishE4Code(value);
        }
        if (head === 0) written[target] += 1;
      }
    }
  }
  for (let token = 0; token < tokens; ++token) {
    if (written[token] !== 1) throw new Error(`oracleWindowAttention: token ${token} written ${written[token]} times`);
  }
  return output;
}

// ---------------------------------------------------------------------------------------------------------------
// ViT (shaders/vit.wgsl).
// ---------------------------------------------------------------------------------------------------------------

/** `tree_sum16`: halves of 16 pair sums folded 8 -> 4 -> 2 -> 1, every add rounded to half. */
function treeSum16(r: ArrayLike<number>): number {
  const s8: number[] = [];
  for (let c = 0; c < 8; ++c) s8.push(roundF16(r[c] + r[c + 8]));
  const s4: number[] = [];
  for (let c = 0; c < 4; ++c) s4.push(roundF16(s8[c] + s8[c + 4]));
  return roundF16(roundF16(s4[0] + s4[2]) + roundF16(s4[1] + s4[3]));
}

/** `vit_norm`: pair squares summed in f32 (high square rounded first) and rounded once, tree, f16(1/sqrt). */
export function vitNorm(values: ArrayLike<number>): number {
  const r: number[] = [];
  for (let c = 0; c < 16; ++c) {
    const highSquare = roundF16(values[c + 16] * values[c + 16]);
    r.push(roundF16(fround(values[c] * values[c] + highSquare)));
  }
  return roundF16(fround(1 / fround(Math.sqrt(treeSum16(r)))));
}

/**
 * `vit_normalize`: E4M3 bytes `[tokens][heads * 96]` from raw qkv halves `[tokens][heads * 96]` and per-head f32
 * learned scales. (The tensor it writes has `paddedTokens` rows; the padding rows are never written.)
 */
export function oracleVitNormalize({
  qkv,
  scales,
  tokens,
  heads,
}: {
  qkv: Uint16Array;
  scales: Float32Array;
  tokens: number;
  heads: number;
}): Uint8Array {
  const stride = heads * 96;
  const output = new Uint8Array(tokens * stride);
  const headScale = roundF16(fround(Math.sqrt(32)));
  for (let token = 0; token < tokens; ++token) {
    for (let head = 0; head < heads; ++head) {
      const base = token * stride + head * 96;
      const q: number[] = [];
      const k: number[] = [];
      for (let c = 0; c < 32; ++c) {
        q.push(f16ToNumber(qkv[base + c]));
        k.push(f16ToNumber(qkv[base + 32 + c]));
      }
      const qNorm = vitNorm(q);
      const kNorm = vitNorm(k);
      const learned = roundF16(scales[head]);
      for (let c = 0; c < 32; ++c) {
        const nq = roundF16(roundF16(roundF16(q[c] * qNorm) * headScale) * learned);
        output[base + c] = publishE4Code(nq);
        output[base + 32 + c] = publishE4Code(roundF16(k[c] * kNorm));
        output[base + 64 + c] = e4m3FromF16Bits(qkv[base + 64 + c]);
      }
    }
  }
  return output;
}

/**
 * `vit_attend`: E4M3 bytes `[tokens][heads * 32]` from the normalized tensor `[paddedTokens][heads * 96]` (rows past
 * `tokens` are the padding; the network keeps them zero).
 */
export function oracleVitAttend({
  normalized,
  tokens,
  heads,
  paddedTokens,
}: {
  normalized: Uint8Array;
  tokens: number;
  heads: number;
  paddedTokens: number;
}): Uint8Array {
  const stride = heads * 96;
  const channels = heads * 32;
  if (normalized.length < paddedTokens * stride) throw new Error('oracleVitAttend: normalized too short');
  const output = new Uint8Array(tokens * channels);
  const at = (index: number): number => e4m3ToNumber(normalized[index]);
  for (let head = 0; head < heads; ++head) {
    for (let token = 0; token < tokens; ++token) {
      const query: number[] = [];
      for (let c = 0; c < 32; ++c) query.push(at(token * stride + head * 96 + c));
      const scores = new Float64Array(paddedTokens);
      for (let key = 0; key < paddedTokens; ++key) {
        let score = 0;
        for (let step = 0; step < 2; ++step) {
          const b: number[] = [];
          for (let i = 0; i < 16; ++i) b.push(at(key * stride + head * 96 + 32 + step * 16 + i));
          score = adaFp8Fdpa16(query.slice(step * 16, step * 16 + 16), b, 16, score);
        }
        scores[key] = vitExpWeight(score);
      }
      const pair = (base: number, p: number, parity: number): number => {
        const key = base + p * 2 + parity;
        const a = roundF16(scores[key] + scores[key + 8]);
        const b = roundF16(scores[key + 16] + scores[key + 24]);
        const c = roundF16(scores[key + 32] + scores[key + 40]);
        const d = roundF16(scores[key + 48] + scores[key + 56]);
        return roundF16(roundF16(roundF16(a + b) + c) + d);
      };
      const softmax64 = (base: number): number => {
        const half = (parity: number): number => {
          const e0 = roundF16(pair(base, 0, parity) + pair(base, 1, parity));
          const e1 = roundF16(e0 + pair(base, 2, parity));
          return roundF16(e1 + pair(base, 3, parity));
        };
        return roundF16(half(0) + half(1));
      };
      let total = 0;
      for (let base = 0; base < paddedTokens; base += 64) total = roundF16(total + softmax64(base));
      const padding = paddedTokens - tokens;
      if (padding > 0) {
        const correction = roundF16(fround(vitExpWeight(0) * padding));
        total = roundF16(total - correction);
      }
      const reciprocal = roundF16(fround(1 / total));
      const weights = Array.from(scores, publishFp8);
      for (let component = 0; component < 32; ++component) {
        let value = 0;
        for (let k0 = 0; k0 < paddedTokens; k0 += 16) {
          const b: number[] = [];
          for (let i = 0; i < 16; ++i) b.push(at((k0 + i) * stride + head * 96 + 64 + component));
          value = adaFp8Fdpa16(weights.slice(k0, k0 + 16), b, 16, value);
        }
        output[token * channels + head * 32 + component] = publishE4Code(roundF16(fround(value * reciprocal)));
      }
    }
  }
  return output;
}
