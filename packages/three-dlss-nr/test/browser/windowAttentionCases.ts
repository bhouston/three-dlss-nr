// Cases and oracle answers for the Chrome reference harness (windowAttentionChrome.mjs), bundled by esbuild there.
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Uses exactly the inputs of
// src/kernels/windowAttention.gpu.test.ts, so a pass here plus that file's oracle tests tie our TSL to the reference.

import { windowPhase } from '../../src/geometry.js';
import { oracleWindowAttention } from '../oracle/attention.js';
import { makePrior, makeQkv, makeScales } from '../oracle/attentionInputs.js';

export interface ChromeWindowCase {
  name: string;
  width: number;
  height: number;
  heads: number;
  phase: number;
  shiftX: number;
  shiftY: number;
  /** Little-endian bytes, base64. */
  qkv: string;
  prior: string;
  scales: string;
  /** Oracle output bytes `[width * height][heads * 32]`, base64. */
  expected: string;
}

const base64 = (view: ArrayBufferView): string =>
  Buffer.from(view.buffer, view.byteOffset, view.byteLength).toString('base64');

/** The window attention test cases (same shapes, phases, seeds and edge-case mixes as the GPU tests). */
export function windowCases(): ChromeWindowCase[] {
  const shapes: [number, number, number, number, number, number?, number?][] = [
    [20, 12, 1, 0, 1],
    [20, 12, 2, 1, 2],
    [20, 12, 1, 2, 3],
    [20, 12, 2, 3, 4],
    [44, 40, 1, 0, 5],
    [44, 40, 2, 1, 6],
    [44, 40, 1, 2, 7],
    [44, 40, 1, 3, 8],
    [24, 16, 16, 1, 9],
    [24, 16, 16, 2, 10],
    [24, 16, 2, 1, 21, 0.3, 0],
    [24, 16, 2, 3, 22, 0, 0.4],
    [20, 12, 1, 0, 23, 0.25, 0.25],
  ];
  return shapes.map(([width, height, heads, phase, seed, zeroTokens, tinyTokens]) => {
    const qkv = makeQkv(width * height, heads, { seed, zeroTokens, tinyTokens });
    const prior = makePrior(heads, seed);
    const scales = makeScales(heads, seed, 2, 6);
    const [shiftX, shiftY] = windowPhase(phase);
    const expected = oracleWindowAttention({ width, height, heads, shiftX, shiftY, qkv, prior, scales });
    return {
      name: `${width}x${height} heads ${heads} phase ${phase} seed ${seed}`,
      width,
      height,
      heads,
      phase,
      shiftX,
      shiftY,
      qkv: base64(qkv),
      prior: base64(prior),
      scales: base64(scales),
      expected: base64(expected),
    };
  });
}
