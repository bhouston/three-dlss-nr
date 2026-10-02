// The TSL kernels port the reference's *composed* production WGSL (design decision 1): the FP8 GEMM variants of
// src/matmul/variants.js and the window attention of src/window/variants.js (OpenDLSS-NR WebGPU port, maan, MIT).
// This pins a hash of every composed text, so a submodule bump that changes what we port fails here first.
// To inspect a variant: node -e "import('<ref>/src/matmul/variants.js').then(m => console.log(m.variantCode({...})))".

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';

import { variantCode } from '@ref/matmul/variants.js';
import { windowAttentionCode } from '@ref/window/variants.js';

const sha256 = (text: string): string => createHash('sha256').update(text).digest('hex').slice(0, 16);

describe('composed reference WGSL', () => {
  it('FP8 GEMM variants (output x residual x batched, tile128)', () => {
    const hashes: Record<string, string> = {};
    for (const output of ['e4', 'half', 'dual'] as const) {
      for (const residual of ['e4', 'half'] as const) {
        for (const batched of [false, true]) {
          hashes[`${output}/${residual}${batched ? '/batched' : ''}`] = sha256(
            variantCode({ output, residual, batched, tile128: true }),
          );
        }
      }
    }
    expect(hashes).toMatchSnapshot();
  });

  it('window attention', () => {
    expect(sha256(windowAttentionCode())).toMatchSnapshot();
  });
});
