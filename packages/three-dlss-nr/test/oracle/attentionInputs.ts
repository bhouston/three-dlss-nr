// Deterministic inputs for the attention tests. Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT).
//
// Raw qkv halves are approximately Gaussian with a per-token gain (so norms and scores vary), sprinkled with the
// values the kernels treat specially: +0, -0, tiny halves (including subnormals) and whole tokens of zeros or of tiny
// values (the 0 * inf and overflowing-norm paths). Priors are halves in [-1, 1], window scales f32 in [2, 6], ViT
// learned scales in [0.15, 0.5] - the ranges of the synthetic model (design 5.1).

import { f16Bits, Xorshift } from '../../src/numerics/oracle.js';

export interface QkvOptions {
  seed: number;
  /** Fraction of tokens whose q and k are all +-0. */
  zeroTokens?: number;
  /** Fraction of tokens whose q and k are tiny (|x| ~ 2^-14 .. 2^-20, the norm overflows half). */
  tinyTokens?: number;
}

/** A uniform draw in [0, 1). */
const unit = (rng: Xorshift): number => rng.next() / 2 ** 32;

/** Approximately N(0, 1) (Irwin-Hall of four uniforms, centred and scaled). */
const gaussian = (rng: Xorshift): number => (unit(rng) + unit(rng) + unit(rng) + unit(rng) - 2) * Math.sqrt(3);

/** Raw qkv half bit patterns `[tokens][heads * 96]`. */
export function makeQkv(
  tokens: number,
  heads: number,
  { seed, zeroTokens = 0.03, tinyTokens = 0.02 }: QkvOptions,
): Uint16Array {
  const rng = new Xorshift(seed * 2654435761 + 1);
  const stride = heads * 96;
  const halves = new Uint16Array(tokens * stride);
  for (let token = 0; token < tokens; ++token) {
    for (let head = 0; head < heads; ++head) {
      const kind = unit(rng);
      const gain = 2 ** (unit(rng) * 6 - 3);
      for (let c = 0; c < 96; ++c) {
        const index = token * stride + head * 96 + c;
        const isQk = c < 64;
        let value: number;
        if (isQk && kind < zeroTokens) value = unit(rng) < 0.5 ? 0 : -0;
        else if (isQk && kind < zeroTokens + tinyTokens) value = gaussian(rng) * 2 ** -16;
        else {
          const special = unit(rng);
          if (special < 0.03) value = 0;
          else if (special < 0.05) value = -0;
          else if (special < 0.07) value = gaussian(rng) * 2 ** -15;
          else value = gaussian(rng) * gain;
        }
        halves[index] = f16Bits(value);
      }
    }
  }
  return halves;
}

/** Prior half bit patterns `[heads][64][64]`, uniform in [-1, 1] (with a few -0 and +0). */
export function makePrior(heads: number, seed: number): Uint16Array {
  const rng = new Xorshift(seed * 40503 + 7);
  return Uint16Array.from({ length: heads * 4096 }, () => {
    const u = unit(rng);
    if (u < 0.01) return 0x8000;
    if (u < 0.02) return 0;
    return f16Bits(unit(rng) * 2 - 1);
  });
}

/** Per-head f32 scales uniform in [lo, hi). */
export function makeScales(heads: number, seed: number, lo: number, hi: number): Float32Array {
  const rng = new Xorshift(seed * 69069 + 3);
  return Float32Array.from({ length: heads }, () => lo + unit(rng) * (hi - lo));
}
