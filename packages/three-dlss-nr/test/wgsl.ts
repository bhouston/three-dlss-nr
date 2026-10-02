// Assertions over the WGSL three generates for a kernel (`kernelWGSL`). Part of three-dlss-nr (a port to three.js of
// OpenDLSS-NR by maan, MIT).

import { expect } from 'vitest';

/**
 * WGSL with three's per-build identifiers normalized (`nodeVar12` -> `nodeVar#`, `NodeBuffer_34` -> `NodeBuffer_#`),
 * for snapshots that should only change when the kernel's code changes (design section 3, codegen drift).
 */
export function normalizeWGSL(wgsl: string): string {
  return wgsl
    .replace(/\bnodeVar\d+\b/g, 'nodeVar#')
    .replace(/\bNodeBuffer_\d+/g, 'NodeBuffer_#')
    .replace(/\bnodeUniform\d+\b/g, 'nodeUniform#')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n');
}

/**
 * R2: TSL compares mismatched operand types as f32 (`f32(index) < f32(rows)`), which is wrong above 2^24. Fails if a
 * comparison has an `f32(...)` conversion as an operand. (Float comparisons of float values are not flagged.)
 */
export function expectIntegerComparisons(wgsl: string): void {
  const offending = wgsl
    .split('\n')
    .filter((line) => /(?:[<>]=?|[!=]=)\s*f32\s*\(|f32\s*\([^()]*\)\s*(?:[<>]=?|[!=]=)/.test(line));
  expect(offending, `f32 conversions in comparisons (R2):\n${offending.join('\n')}`).toEqual([]);
}

/** R4/R5: functions the published arithmetic must not use (they are approximations or fuse). */
export function expectNoApproximations(wgsl: string): void {
  const banned = /\b(fma|exp2|log2|exp|log|pow|inverseSqrt|mix|smoothstep|pack2x16float)\s*\(/g;
  const found = [...wgsl.matchAll(banned)].map((match) => match[1]);
  expect(found, `banned functions in published arithmetic: ${found.join(', ')}`).toEqual([]);
}
