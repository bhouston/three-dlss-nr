// Synthetic input features: deterministic (pinned hash), on the half grid, preprocess.wgsl's lane layout and mirror
// (OpenDLSS-NR's ports/browser-webgpu/shaders/preprocess.wgsl, by maan, MIT).

import { describe, expect, it } from 'vitest';

import { geometryFromValid } from '../geometry.js';
import { sha256Hex } from '../model/manifest.js';
import { roundF16 } from '../numerics/oracle.js';
import { centredProxy, FEATURE_LANES, syntheticFeatures } from './features.js';

const geometry = geometryFromValid(64, 64);
const features = syntheticFeatures(geometry);
const lane = (x: number, y: number, l: number) => features[(y * geometry.fullWidth + x) * FEATURE_LANES + l];

describe('synthetic features', () => {
  it('cover the padded field, 16 lanes per pixel, and are pinned', async () => {
    expect(features.length).toBe(336 * 320 * 16);
    expect(await sha256Hex(new Uint8Array(features.buffer))).toMatchInlineSnapshot(
      `"7e6610862e67383aaccd320b5c6c5ce47fefcab6bb3de269a1f1dd0a524ccf9d"`,
    );
    expect(syntheticFeatures(geometry)).toEqual(features);
    expect(syntheticFeatures(geometry, { seed: 2 })).not.toEqual(features);
  });

  it('put every value on the half grid, with the reference demo constants in lanes 3 and 10-15', () => {
    expect(features.filter((value) => roundF16(value) !== value)).toHaveLength(0);
    for (const [l, value] of [
      [3, 1],
      [10, 0],
      [11, 1],
      [12, 1],
      [13, -1],
      [14, -1],
      [15, 0],
    ]) {
      for (let p = 0; p < geometry.fullRows; p += 97) expect(features[p * 16 + l]).toBe(value);
    }
  });

  it('carry unit-variance noise and a centred proxy within +-1/16', () => {
    let sum = 0;
    let squares = 0;
    let outside = 0;
    for (let p = 0; p < geometry.fullRows; ++p) {
      for (let l = 0; l < 3; ++l) {
        sum += features[p * 16 + l];
        squares += features[p * 16 + l] ** 2;
      }
      for (let l = 4; l < 10; ++l) if (Math.abs(features[p * 16 + l]) > 0.0625) outside += 1;
    }
    expect(outside).toBe(0);
    const count = geometry.fullRows * 3;
    expect(Math.abs(sum / count)).toBeLessThan(0.01);
    expect(squares / count).toBeGreaterThan(0.95);
    expect(squares / count).toBeLessThan(1.05);
    expect(centredProxy(1)).toBe(0.0625);
    expect(centredProxy(0)).toBe(-0.0625);
  });

  it('mirror the source outside the valid image while the history is a shifted copy', () => {
    for (const y of [0, 10, 63]) {
      for (const x of [64, 70, 100]) {
        for (let l = 4; l < 7; ++l) expect(lane(x, y, l)).toBe(lane(2 * 64 - x - 2, y, l));
      }
    }
    for (let l = 4; l < 7; ++l) expect(lane(30, 64, l)).toBe(lane(30, 62, l));
    // History lanes = the proxy at (x - 2, y - 1).
    for (const [x, y] of [
      [10, 10],
      [40, 33],
    ]) {
      for (let c = 0; c < 3; ++c) expect(lane(x, y, 7 + c)).toBe(lane(x - 2, y - 1, 4 + c));
    }
  });
});
