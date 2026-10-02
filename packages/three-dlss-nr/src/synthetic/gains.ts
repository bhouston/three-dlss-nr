// The constants of the synthetic weights.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). No NVIDIA weights are included anywhere in this
// project: tests and the demo run on these deterministic synthetic weights, in the reference's model directory layout.
//
// The values keep activations in a useful range through all 71 blocks (contractive residual stream, softmaxes that
// are neither one-hot nor flat). Change them only together with the calibration gate
// (`syntheticStats.gpu.test.ts`) and the pinned stage hashes (`generate.test.ts`).

import type { Fp8Role } from '../model/layouts.js';

/** Gain `g` of an FP8 matrix: `w = e4m3(g * N(0, 1) / sqrt(fanIn))`, fan-in = one batch's K. */
export const SYNTHETIC_GAINS: Readonly<Record<Fp8Role, number>> = {
  expand: 1.0,
  transition: 1.0,
  contract: 0.8,
  merge: 0.8,
  qkv: 0.8,
  projection: 0.8,
};

/** Fraction of FP8 weights forced to exact zero (exercises the zero skip of the shared exponent): 4 % of 2^32. */
export const SYNTHETIC_ZERO_THRESHOLD = 171798692;

/** Half-open uniform ranges `[low, high)` of the scalar parameters. */
export const SYNTHETIC_RANGES = {
  /** Every per-channel f16 skip / transition / post-blend scale. */
  skipScale: [0.5, 0.95],
  /** Window attention per-head f32 scale. */
  windowScale: [2, 6],
  /** ViT learned per-head f32 scale (times sqrt(32) in the kernel). */
  vitScale: [0.15, 0.5],
  /** Window attention prior (f16). */
  prior: [-1, 1],
} as const satisfies Record<string, readonly [number, number]>;

/** Standard deviations of the two f16 matrices. */
export const SYNTHETIC_F16_SIGMA = { adapter: 0.35, head: 0.1 } as const;

/** The f16 constants: the temporal blend scale and the unread ViT `layer3`. */
export const SYNTHETIC_CONSTANTS = { 'blend-scale': 0.75, 'vit-layer3': 1.0 } as const;
