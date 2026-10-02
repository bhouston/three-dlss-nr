// The geometry port against the reference's ports/browser-webgpu/src/geometry.js (OpenDLSS-NR, maan, MIT).

import { describe, expect, it } from 'vitest';

import * as ref from '@ref/geometry.js';

import {
  fusedLayout,
  geometryFromValid,
  grid1d,
  postFusedLayout,
  preFusedLayout,
  upsampleFusedLayout,
  windowPhase,
  WindowPhases,
} from './geometry.js';

const lcg = (seed: number) => () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0);

/** 200 sizes: every edge class (below the 320 floor, around alignment steps, large, invalid). */
function sizes(): [number, number][] {
  const out: [number, number][] = [];
  const next = lcg(7);
  const fixed = [
    [64, 64],
    [512, 512],
    [1280, 720],
    [1920, 1080],
    [33, 33],
    [32, 32],
    [256, 256],
    [257, 300],
    [3840, 2160],
    [40, 900],
  ];
  for (const size of fixed) out.push(size as [number, number]);
  while (out.length < 200) out.push([8 + (next() % 2600), 8 + (next() % 1600)]);
  return out;
}

const outcome = <T>(f: () => T): T | { error: string } => {
  try {
    return f();
  } catch (error) {
    return { error: (error as Error).message };
  }
};

describe('geometry vs the reference', () => {
  it('geometryFromValid for 200 sizes (including the ones it rejects)', () => {
    for (const [w, h] of sizes()) {
      expect(
        outcome(() => geometryFromValid(w, h)),
        `${w}x${h}`,
      ).toEqual(outcome(() => ref.geometryFromValid(w, h)));
    }
  });

  it('documented geometries (design 1.3)', () => {
    expect(geometryFromValid(64, 64)).toMatchObject({
      fullWidth: 336,
      fullHeight: 320,
      vitTokens: 64,
      paddedVitTokens: 64,
    });
    expect(geometryFromValid(512, 512)).toMatchObject({
      fullWidth: 576,
      fullHeight: 512,
      vitTokens: 96,
      paddedVitTokens: 128,
    });
    expect(geometryFromValid(1280, 720)).toMatchObject({
      fullWidth: 1344,
      fullHeight: 768,
      vitTokens: 288,
      paddedVitTokens: 320,
    });
  });

  it('the four fused layouts', () => {
    for (const channels of [32, 64, 128, 256]) {
      expect(fusedLayout(channels)).toEqual(ref.fusedLayout(channels));
      expect(fusedLayout(channels, 4096)).toEqual(ref.fusedLayout(channels, 4096));
      expect(upsampleFusedLayout(channels * 2, channels)).toEqual(ref.upsampleFusedLayout(channels * 2, channels));
    }
    expect(preFusedLayout()).toEqual(ref.preFusedLayout());
    expect(postFusedLayout()).toEqual(ref.postFusedLayout());
    expect(() => fusedLayout(512)).toThrow();
    expect(() => upsampleFusedLayout(64, 64)).toThrow();
    // The byte lengths the synthetic model is built from (design 5.1).
    expect(preFusedLayout().endWithoutPadding + 16).toBe(21696);
    expect([32, 64, 128, 256].map((c) => fusedLayout(c).endWithoutPadding + 16)).toEqual([
      20672, 61760, 197184, 689232,
    ]);
    expect(postFusedLayout().endWithoutPadding).toBe(21808);
  });

  it('window phases cycle per level', () => {
    for (let index = 0; index < 12; ++index) expect(windowPhase(index)).toEqual(ref.windowPhase(index));
    const ours = new WindowPhases();
    const theirs = new ref.WindowPhases();
    for (const level of [6, 0, 0, 1, 6, 5, 0]) expect(ours.take(level)).toBe(theirs.take(level));
    ours.reset();
    expect(ours.take(0)).toBe(0);
  });

  it('grid1d folds past 65535 workgroups as graph.js does', () => {
    expect(grid1d(64)).toEqual([1, 1, 1]);
    expect(grid1d(65535 * 64)).toEqual([65535, 1, 1]);
    expect(grid1d(65536 * 64)).toEqual([65535, 2, 1]);
    expect(grid1d(1920 * 1152 * 16)).toEqual([65535, Math.ceil((1920 * 1152 * 16) / 64 / 65535), 1]);
  });
});
