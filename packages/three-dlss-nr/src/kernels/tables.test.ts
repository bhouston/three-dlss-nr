// The FP8 GEMM's lookup tables against the reference's (OpenDLSS-NR WebGPU port by maan, MIT).
//
// The SiLU table is also compared with the reference's GPU-built `packedSiluTable` on real f16 hardware by
// test/browser/run-fp8-gemm-chrome.mjs (equal for every non-NaN half).

import { describe, expect, it } from 'vitest';

import { weightMetadataWords } from '@ref/matmul/weight-table.js';

import {
  e4m3FromF16Bits,
  f16Bits,
  f16ToNumber,
  f32FromBits,
  mpCubicSilu,
  siluE4CodeTable,
} from '../numerics/oracle.js';
import { attributeBytes } from '../tensors.js';
import {
  E4_OPERAND_EXPONENT_BIAS,
  e4OperandTable,
  e4OperandTableAttribute,
  siluCodeTable,
  siluCodeTableAttribute,
} from './tables.js';

describe('GEMM tables', () => {
  it('the SiLU code table is the oracle table, uploaded byte for byte', () => {
    const table = siluCodeTable();
    expect(table).toEqual(siluE4CodeTable());
    expect(attributeBytes(siluCodeTableAttribute())).toEqual(table);
  });

  it('a zero activation publishes 0x00 through SiLU; a tiny negative one 0x80', () => {
    const table = siluCodeTable();
    expect(table[0x0000]).toBe(0x00);
    expect(table[0x8000]).toBe(0x00); // -0: fp8_domain returns +0 for a zero input
    expect(table[0x8001]).toBe(0x80); // -2^-24 activates to a tiny negative that quantizes to -0
    expect(table[0x3c00]).toBe(e4m3FromF16Bits(f16Bits(mpCubicSilu(1)))); // SiLU(1) published
  });

  it("the operand table decodes like the reference's weight metadata (value x 4 as a half, exponent)", () => {
    const metadata: Uint32Array = weightMetadataWords();
    const table = e4OperandTable();
    for (let code = 0; code < 256; ++code) {
      const value = f32FromBits((table[code] & 0xfff00000) >>> 0);
      const exponent = (table[code] & 0xff) - E4_OPERAND_EXPONENT_BIAS;
      expect(value * 4, `code ${code} value`).toBe(f16ToNumber(metadata[code] & 0xffff));
      expect(exponent, `code ${code} exponent`).toBe(f16ToNumber(metadata[code] >>> 16));
    }
    expect(attributeBytes(e4OperandTableAttribute())).toEqual(new Uint8Array(table.buffer));
  });
});
