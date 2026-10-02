// CPU oracles of the reference's two GEMMs, for tests.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Semantics of the composed production FP8
// GEMM of ports/browser-webgpu/src/matmul/ (`variantCode({ output, residual, batched, tile128: true })`, design
// Appendix A.1) and of shaders/gemm_f16.wgsl (A.2), written over the TS oracle in src/numerics/oracle.ts. Slow and
// literal; meant for small shapes.

import type { GemmSpec } from '../../src/types.js';
import {
  Xorshift,
  adaF16Fdpa8,
  e4m3Exponent,
  e4m3FromF16Bits,
  e4m3ToNumber,
  f16Bits,
  f16Exponent,
  f16ToNumber,
  packedInputIndex,
  packedWeightIndex,
  publishE4CodeGemm,
  roundF16,
  siluE4CodeTable,
} from '../../src/numerics/oracle.js';

let siluCodes: Uint8Array | null = null;

/**
 * One 16-product group of the composed GEMM (`quad_fdpa_16_*`): shared exponent over the accumulator (floor -14,
 * -21 when zero) and every pair with both operands nonzero; aligned truncated terms summed exactly; one rounding to
 * half; a zero result is +0. Unlike numerics.wgsl's `ada_fp8_fdpa16`, a non-finite accumulator is not passed through.
 */
export function gemmGroup(a: ArrayLike<number>, b: ArrayLike<number>, accumulator: number): number {
  let maximumExponent = accumulator !== 0 ? f16Exponent(accumulator) : -21;
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== 0 && b[i] !== 0) maximumExponent = Math.max(maximumExponent, e4m3Exponent(a[i]) + e4m3Exponent(b[i]));
  }
  const scale = 2 ** (13 - maximumExponent);
  let units = Math.trunc(accumulator * scale);
  for (let i = 0; i < a.length; ++i) units += Math.trunc(a[i] * b[i] * scale);
  const rounded = roundF16(units * 2 ** (maximumExponent - 13));
  return rounded === 0 ? 0 : rounded;
}

export interface OracleGemmFp8Args {
  rows: number;
  /** Input channels of one matrix. */
  k: number;
  /** Output channels of one matrix. */
  n: number;
  batches?: number;
  broadcast?: boolean;
  partition?: 0 | 256 | 512 | 1024;
  silu?: boolean;
  output: 'e4' | 'half' | 'dual';
  /** E4 activation bytes `[rows][inputChannels]`, channel `k` of a batch at `packedInputIndex(k)`. */
  input: Uint8Array;
  /** Row stride of `input` (defaults to `k`, or `k * batches` when batched and not broadcast). */
  inputChannels?: number;
  /** `batches` packed matrices of `k x n` bytes back to back (`packedWeightIndex` order). */
  weights: Uint8Array;
  /** Skip tensor at the output index: E4 bytes or half bit patterns. */
  residual?: { format: 'e4' | 'f16'; data: Uint8Array | Uint16Array };
  /** Per-column skip scales (half bit patterns, indexed by the column within a batch). */
  scale?: Uint16Array;
  /** Row stride of the output (defaults to `n * batches`). */
  outputChannels?: number;
}

export interface OracleGemmResult {
  /** E4 codes `[rows][outputChannels]` (outputs `e4`, `dual`). */
  e4?: Uint8Array;
  /** Half bit patterns `[rows][outputChannels]` (outputs `half`, `dual`). */
  half?: Uint16Array;
  /** Final accumulator values (halves as numbers), `[rows][outputChannels]`. */
  values: Float64Array;
}

/** The composed FP8 GEMM (design Appendix A.1). */
export function oracleGemmFp8(args: OracleGemmFp8Args): OracleGemmResult {
  const { rows, k, n, output } = args;
  const batches = args.batches ?? 1;
  const broadcast = args.broadcast ?? false;
  const partition = args.partition ?? 0;
  const inputChannels = args.inputChannels ?? (batches > 1 && !broadcast ? k * batches : k);
  const outputChannels = args.outputChannels ?? n * batches;
  if (k % 32 || n % 16) throw new Error('oracleGemmFp8: K must be a multiple of 32 and N of 16');
  if (args.silu && output !== 'e4') throw new Error('oracleGemmFp8: SiLU publishes E4 only');
  const values = new Float64Array(rows * outputChannels);
  const a = new Float64Array(16);
  const b = new Float64Array(16);
  for (let batch = 0; batch < batches; ++batch) {
    for (let row = 0; row < rows; ++row) {
      for (let column = 0; column < n; ++column) {
        const outputIndex = row * outputChannels + batch * n + column;
        let seed = 0;
        if (args.residual) {
          const { format, data } = args.residual;
          const residual = format === 'e4' ? e4m3ToNumber(data[outputIndex]) : f16ToNumber(data[outputIndex]);
          const scale = args.scale ? f16ToNumber(args.scale[column]) : 1;
          seed = roundF16(Math.fround(residual * scale));
          if (seed === 0) seed = 0; // `0.0 + r` turns -0 into +0
        }
        let sums = seed;
        let part = 0;
        const span = partition || 1024;
        for (let kBase = 0; kBase < k; kBase += 32) {
          for (let group = 0; group < 2; ++group) {
            for (let i = 0; i < 16; ++i) {
              const kk = kBase + group * 16 + i;
              const inputBase = row * inputChannels + (broadcast ? 0 : batch * k);
              a[i] = e4m3ToNumber(args.input[inputBase + packedInputIndex(kk)]);
              b[i] = e4m3ToNumber(args.weights[batch * k * n + packedWeightIndex(kk, column, n)]);
            }
            sums = gemmGroup(a, b, sums);
          }
          if (partition && ((kBase + 32) % span === 0 || kBase + 32 >= k)) {
            part = kBase < span ? sums : roundF16(part + sums);
            sums = 0;
          }
        }
        values[outputIndex] = partition ? part : sums;
      }
    }
  }
  const result: OracleGemmResult = { values };
  if (output !== 'half') {
    if (args.silu) siluCodes ??= siluE4CodeTable();
    const e4 = new Uint8Array(rows * outputChannels);
    for (let i = 0; i < values.length; ++i) {
      e4[i] = args.silu ? siluCodes![f16Bits(values[i])] : publishE4CodeGemm(values[i]);
    }
    result.e4 = e4;
  }
  if (output !== 'e4') result.half = Uint16Array.from(values, (value) => f16Bits(value));
  return result;
}

export interface OracleGemmF16Args {
  rows: number;
  k: number;
  n: number;
  /** Half bit patterns `[rows][inputChannels]`. */
  input: Uint16Array;
  inputChannels?: number;
  /** Half bit patterns `[k][paddedN]`. */
  weights: Uint16Array;
  paddedN: number;
  outputChannels?: number;
}

/** shaders/gemm_f16.wgsl: per output, `ada_f16_fdpa8` over K in steps of 8 from a zero accumulator. */
export function oracleGemmF16(args: OracleGemmF16Args): { f32: Float32Array; half: Uint16Array; e4: Uint8Array } {
  const { rows, k, n, paddedN } = args;
  const inputChannels = args.inputChannels ?? k;
  const outputChannels = args.outputChannels ?? n;
  const f32 = new Float32Array(rows * outputChannels);
  const half = new Uint16Array(rows * outputChannels);
  const e4 = new Uint8Array(rows * outputChannels);
  const a = new Float64Array(8);
  const b = new Float64Array(8);
  for (let row = 0; row < rows; ++row) {
    for (let column = 0; column < n; ++column) {
      let value = 0;
      for (let base = 0; base < k; base += 8) {
        for (let i = 0; i < 8; ++i) {
          a[i] = f16ToNumber(args.input[row * inputChannels + base + i]);
          b[i] = f16ToNumber(args.weights[(base + i) * paddedN + column]);
        }
        value = adaF16Fdpa8(a, b, 8, value);
      }
      const index = row * outputChannels + column;
      f32[index] = value;
      half[index] = f16Bits(value);
      e4[index] = e4m3FromF16Bits(half[index]);
    }
  }
  return { f32, half, e4 };
}

// ---------------------------------------------------------------------------------------------------------------
// The graph's FP8 GEMM roles and deterministic inputs for them (shared by the GEMM tests and the browser harness).
// ---------------------------------------------------------------------------------------------------------------

/** One `Graph.gemm` call shape. `inputChannels` defaults to `k`, or `k * batches` when batched and not broadcast. */
export interface GemmRole {
  label: string;
  rows: number;
  k: number;
  n: number;
  batches?: number;
  broadcast?: boolean;
  partition?: 0 | 256 | 512 | 1024;
  silu?: boolean;
  output: 'e4' | 'half' | 'dual';
  residual?: 'e4' | 'f16';
  inputChannels?: number;
}

/** The graph's GEMM roles (graph.js:181-526), at 50 rows (not a multiple of 32) unless noted. */
export const GEMM_ROLES: GemmRole[] = [
  // Dense 32-channel blocks (0-4, 66-70): expand (SiLU), contract (dual, f16 skip), qkv, projection (dual, f16 skip).
  { label: 'dense expand 32->128 SiLU', rows: 50, k: 32, n: 128, silu: true, output: 'e4' },
  { label: 'dense contract 128->32 dual, f16 skip', rows: 50, k: 128, n: 32, output: 'dual', residual: 'f16' },
  { label: 'qkv 32->96 half', rows: 50, k: 32, n: 96, output: 'half' },
  { label: 'projection 32 dual, f16 skip', rows: 50, k: 32, n: 32, output: 'dual', residual: 'f16' },
  { label: 'projection 32 E4, f16 skip', rows: 96, k: 32, n: 32, output: 'e4', residual: 'f16' },
  // Expert blocks (C = 64: 2 experts; C = 256: 8 experts).
  {
    label: 'expert expand 64 (2x128, broadcast, SiLU)',
    rows: 50,
    k: 64,
    n: 128,
    batches: 2,
    broadcast: true,
    silu: true,
    output: 'e4',
  },
  { label: 'expert contract 64 (2x 128->32)', rows: 50, k: 128, n: 32, batches: 2, output: 'e4' },
  { label: 'expert merge 64 dual, E4 skip', rows: 50, k: 64, n: 64, output: 'dual', residual: 'e4' },
  { label: 'projection 64 E4, E4 skip', rows: 50, k: 64, n: 64, output: 'e4', residual: 'e4' },
  { label: 'qkv 128->384 half', rows: 32, k: 128, n: 384, output: 'half' },
  {
    label: 'expert expand 256 (8x128, broadcast, SiLU)',
    rows: 32,
    k: 256,
    n: 128,
    batches: 8,
    broadcast: true,
    silu: true,
    output: 'e4',
  },
  { label: 'expert contract 256 (8x 128->32)', rows: 32, k: 128, n: 32, batches: 8, output: 'e4' },
  // The 512 split blocks (23-30, 40-47): 8 branches of 64 -> 256 -> 64.
  { label: 'split layer0 512', rows: 32, k: 512, n: 512, output: 'e4' },
  { label: 'split expand (8x 64->256, SiLU)', rows: 32, k: 64, n: 256, batches: 8, silu: true, output: 'e4' },
  { label: 'split contract (8x 256->64)', rows: 32, k: 256, n: 64, batches: 8, output: 'e4' },
  { label: 'split merge 512, E4 skip', rows: 32, k: 512, n: 512, output: 'e4', residual: 'e4' },
  // Transitions.
  { label: 'transition 4-5 (32->64)', rows: 50, k: 32, n: 64, output: 'e4' },
  { label: 'transition 30-31 (512->1024)', rows: 32, k: 512, n: 1024, output: 'e4' },
  { label: 'decoder transition 128->64 half', rows: 50, k: 128, n: 64, output: 'half' },
  // ViT (partitions; N reduced from 1024 / 3072 / 1024 / 512).
  {
    label: 'ViT contract 4096 (partition 1024, E4 skip)',
    rows: 32,
    k: 4096,
    n: 64,
    partition: 1024,
    output: 'e4',
    residual: 'e4',
  },
  { label: 'ViT qkv 1024 half (partition 512)', rows: 50, k: 1024, n: 96, partition: 512, output: 'half' },
  {
    label: 'ViT projection 1024 (partition 256, E4 skip)',
    rows: 50,
    k: 1024,
    n: 64,
    partition: 256,
    output: 'e4',
    residual: 'e4',
  },
  {
    label: 'transition 38-39 1024->512 half (partition 256)',
    rows: 32,
    k: 1024,
    n: 64,
    partition: 256,
    output: 'half',
  },
];

const fnv1a = (text: string): number => {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; ++i) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return hash;
};

export interface GemmData {
  spec: GemmSpec;
  inputChannels: number;
  outputChannels: number;
  input: Uint8Array;
  weights: Uint8Array;
  scale?: Uint16Array;
  residual?: Uint8Array | Uint16Array;
}

/** Deterministic hostile inputs for a role. */
export function gemmData(role: GemmRole): GemmData {
  const rng = new Xorshift(fnv1a(role.label) | 1);
  const batches = role.batches ?? 1;
  const broadcast = role.broadcast ?? false;
  const inputChannels = role.inputChannels ?? (batches > 1 && !broadcast ? role.k * batches : role.k);
  const outputChannels = role.n * batches;
  const activation = (): number => {
    const draw = rng.next();
    if (draw % 16 === 0) return draw & 0x80; // +-0
    if (draw % 97 === 1) return (draw & 0x80) | 0x7f; // the NaN code reads as 0
    if (draw % 13 === 2) return (draw & 0x80) | ((draw >>> 8) & 0x07); // subnormal
    const code = (draw >>> 8) & 0xff;
    return (code & 0x7f) === 0x7f ? code & 0xfe : code;
  };
  const weight = (): number => {
    const draw = rng.next();
    if (draw % 25 === 0) return draw & 0x80;
    if (draw % 211 === 1) return (draw & 0x80) | 0x7f;
    return (draw & 0x80) | ((draw >>> 8) % 0x52); // |w| <= 9 (model.js)
  };
  const half = (minExponent: number, maxExponent: number): number => {
    const draw = rng.next();
    if (draw % 31 === 0) return draw & 0x8000;
    const exponent = minExponent + ((draw >>> 16) % (maxExponent - minExponent + 1));
    return (draw & 0x8000) | (exponent << 10) | (draw & 0x3ff);
  };
  const data: GemmData = {
    spec: {
      rows: role.rows,
      k: role.k,
      n: role.n,
      batches,
      broadcast,
      partition: role.partition ?? 0,
      silu: role.silu ?? false,
      output: role.output,
      residual: role.residual ? { format: role.residual } : null,
      label: role.label,
    },
    inputChannels,
    outputChannels,
    input: Uint8Array.from({ length: role.rows * inputChannels }, activation),
    weights: Uint8Array.from({ length: batches * role.k * role.n }, weight),
  };
  if (role.residual) {
    // Skip scales in [2^-3, 2); skip values over the whole range down to subnormals, so some products round to -0.
    data.scale = Uint16Array.from({ length: role.n }, () => half(12, 15));
    data.residual =
      role.residual === 'e4'
        ? Uint8Array.from({ length: role.rows * outputChannels }, activation)
        : Uint16Array.from({ length: role.rows * outputChannels }, () => half(0, 20));
  }
  return data;
}
