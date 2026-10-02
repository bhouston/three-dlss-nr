// Deterministic synthetic input features, built on the CPU.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The sixteen lanes follow the reference's
// ports/browser-webgpu/shaders/preprocess.wgsl (`preprocess`) and docs/frame.md: lanes 4-6 are the centred proxy
// `f16(f16(f16(proxy) - 0.5) * 0.125)` with the source mirrored outside the valid image, lanes 7-9 the same for the
// history. Two deliberate differences make the features identical on every machine: the three noise lanes are
// Irwin-Hall normals from an integer hash of the padded coordinate (the reference uses Box-Muller over the GPU's
// approximate transcendentals), and the proxy is a procedural image instead of a recorded one. Both the reference and
// the TSL port then get the same `Float32Array` through `writeFeatures`, so a comparison starts after the only
// hardware-dependent step, as the reference's own fixtures do.

import type { NRGeometry } from '../geometry.js';
import { roundF16 } from '../numerics/oracle.js';
import { irwinHallNormal } from './generate.js';

/** Feature lanes per padded pixel. */
export const FEATURE_LANES = 16;

export interface SyntheticFeaturesOptions {
  /** Noise seed (default 1). */
  seed?: number;
  /** Offset of the history image from the current one, in pixels (default `[2, 1]`). */
  historyShift?: readonly [number, number];
}

/**
 * The procedural proxy code value (0..1, f32) of channel `channel` at valid pixel `(x, y)`: a horizontal gradient
 * (red), a vertical gradient over an 8-pixel checker (green), and a disc with thin grid lines (blue).
 */
export function syntheticProxy(x: number, y: number, channel: number, validWidth: number, validHeight: number): number {
  if (channel === 0) return Math.fround((x + 0.5) / validWidth);
  if (channel === 1) {
    const checker = ((x >> 3) + (y >> 3)) & 1 ? 0.75 : 0.25;
    return Math.fround(((y + 0.5) / validHeight) * 0.5 + checker * 0.5);
  }
  if (x % 16 === 0 || y % 16 === 0) return 1;
  const dx = 2 * x + 1 - validWidth;
  const dy = 2 * y + 1 - validHeight;
  const radius = Math.min(validWidth, validHeight) * 0.66;
  return dx * dx + dy * dy < radius * radius ? 0.875 : 0.125;
}

/** `centred = f16(f16(f16(proxy) - 0.5) * 0.125)`, preprocess.wgsl's rounding chain. */
export const centredProxy = (proxy: number): number => roundF16(roundF16(roundF16(proxy) - 0.5) * 0.125);

/** preprocess.wgsl's mirror at the valid edge (`2 * valid - x - 2`), clamped into the image for tiny sizes. */
const mirror = (coordinate: number, valid: number): number =>
  Math.min(Math.max(coordinate < valid ? coordinate : 2 * valid - coordinate - 2, 0), valid - 1);

/** An integer hash of a padded coordinate and the seed (no transcendentals). */
function coordinateHash(x: number, y: number, seed: number): number {
  let base = (Math.imul(x, 0x8da6b343) ^ Math.imul(seed, 0x9e3779b9) ^ Math.imul(y, 0xd8163841) ^ 0x243f6a88) >>> 0;
  base = ((base >>> ((base >>> 28) + 4)) ^ base) >>> 0;
  base = Math.imul(base, 0x108ef2d9) >>> 0;
  base = ((base >>> 22) ^ base) >>> 0;
  return base || 0x9e3779b9;
}

/**
 * The f32 features `[fullHeight * fullWidth][16]` of one synthetic frame: noise (0-2), 1 (3), centred proxy (4-6),
 * centred history (7-9), then the reference demo's defaults style 0, tone 1, structure 1, -1, -1, 0 (10-15).
 */
export function syntheticFeatures(
  geometry: Pick<NRGeometry, 'validWidth' | 'validHeight' | 'fullWidth' | 'fullHeight'>,
  { seed = 1, historyShift = [2, 1] }: SyntheticFeaturesOptions = {},
): Float32Array {
  const { validWidth, validHeight, fullWidth, fullHeight } = geometry;
  const features = new Float32Array(fullWidth * fullHeight * FEATURE_LANES);
  for (let y = 0; y < fullHeight; ++y) {
    const sy = mirror(y, validHeight);
    const hy = Math.min(Math.max(sy - historyShift[1], 0), validHeight - 1);
    for (let x = 0; x < fullWidth; ++x) {
      const sx = mirror(x, validWidth);
      const hx = Math.min(Math.max(sx - historyShift[0], 0), validWidth - 1);
      const base = (y * fullWidth + x) * FEATURE_LANES;
      let state = coordinateHash(x, y, seed);
      for (let lane = 0; lane < 3; ++lane) {
        state = (state ^ (state << 13)) >>> 0;
        state = (state ^ (state >>> 17)) >>> 0;
        state = (state ^ (state << 5)) >>> 0;
        features[base + lane] = roundF16(irwinHallNormal(state));
      }
      features[base + 3] = 1;
      for (let channel = 0; channel < 3; ++channel) {
        features[base + 4 + channel] = centredProxy(syntheticProxy(sx, sy, channel, validWidth, validHeight));
        features[base + 7 + channel] = centredProxy(syntheticProxy(hx, hy, channel, validWidth, validHeight));
      }
      features[base + 10] = 0;
      features[base + 11] = 1;
      features[base + 12] = 1;
      features[base + 13] = -1;
      features[base + 14] = -1;
      features[base + 15] = 0;
    }
  }
  return features;
}
