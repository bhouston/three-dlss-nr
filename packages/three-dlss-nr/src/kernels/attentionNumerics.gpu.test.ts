// The attention kernels' short publications equal the reference's spelled-out ones bit for bit: `nrRoundHalf` ==
// `f16_to_f32(f16_bits(x))` and `nrPublishE4Value` == `decode_e4m3(encode_e4m3(f16_bits(x)))`, over every half,
// every half midpoint and its f32 neighbours, the overflow edge, and random f32 patterns; checked against the TS
// oracle and against the foundation's TSL `nrRoundF16` / `nrDecodeE4m3(nrEncodeE4m3(nrF16Bits(x)))`.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { uintBitsToFloat, floatBitsToUint } from 'three/tsl';

import { describeMismatches, diffArrays } from '../../test/compare.js';
import { createGpuTestContext, runPerElement, wordsBuffer, type GpuTestContext } from '../../test/gpu.js';
import {
  e4m3FromF16Bits,
  e4m3ToNumber,
  f16Bits,
  f16ToF32Bits,
  f16ToNumber,
  f32Bits,
  f32FromBits,
  Xorshift,
} from '../numerics/oracle.js';
import { nrDecodeE4m3, nrEncodeE4m3, nrF16Bits, nrRoundF16 } from '../tsl/numerics.js';
import { nrPublishE4Value, nrRoundHalf } from './attentionNumerics.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

/** f32 bit patterns covering every rounding case of the half and E4M3 grids. */
function cases(): Uint32Array {
  const out: number[] = [];
  for (let h = 0; h < 0x10000; ++h) {
    const bits = f16ToF32Bits(h);
    out.push(bits, (bits + 1) >>> 0, (bits - 1) >>> 0);
    // The midpoint between h and the next half up (in magnitude), and its f32 neighbours.
    if ((h & 0x7fff) < 0x7c00) {
      const mid = (f16ToNumber(h) + f16ToNumber(h + 1)) / 2;
      if (Number.isFinite(mid)) {
        const m = f32Bits(mid);
        out.push(m, (m + 1) >>> 0, (m - 1) >>> 0);
      }
    }
  }
  // Around the overflow edge (65504 .. 65536) and E4M3's saturation (448 .. 480), both signs.
  for (const edge of [65504, 65520, 65536, 448, 464, 480, 2 ** -14, 2 ** -6, 2 ** -24, 2 ** -25]) {
    const bits = f32Bits(edge);
    for (let d = -4096; d <= 4096; d += 64) out.push((bits + d) >>> 0, ((bits + d) | 0x80000000) >>> 0);
  }
  const rng = new Xorshift(7);
  for (let i = 0; i < 1 << 18; ++i) out.push(rng.next());
  // Normal f32 inputs only: f32 subnormals are flushed by some backends (R10); the kernels never form them.
  return Uint32Array.from(out.filter((bits) => (bits & 0x7f800000) !== 0 || (bits & 0x7fffff) === 0));
}

const run = (label: string, inputs: Uint32Array, fn: (x: any) => any) =>
  runPerElement(gpu.renderer, {
    label,
    count: inputs.length,
    inputs: { x: wordsBuffer(inputs) },
    compute: (index, { x }) => floatBitsToUint(fn(uintBitsToFloat(x.element(index)))),
  });

describe('attention numerics', () => {
  const inputs = cases();

  it('nrRoundHalf == f16_to_f32(f16_bits(x))', async () => {
    const expected = inputs.map((bits) => f16ToF32Bits(f16Bits(f32FromBits(bits))));
    const ours = await run('round half', inputs, (x) => nrRoundHalf(x));
    const vsOracle = diffArrays(ours, expected);
    expect(vsOracle.mismatches, describeMismatches('vs oracle', vsOracle)).toBe(0);
    const theirs = await run('round f16', inputs, (x) => nrRoundF16(x));
    const vsTsl = diffArrays(ours, theirs);
    expect(vsTsl.mismatches, describeMismatches('vs nrRoundF16', vsTsl)).toBe(0);
  });

  it('nrPublishE4Value == decode_e4m3(encode_e4m3(f16_bits(x)))', async () => {
    const expected = inputs.map((bits) => f32Bits(e4m3ToNumber(e4m3FromF16Bits(f16Bits(f32FromBits(bits))))));
    const ours = await run('publish e4', inputs, (x) => nrPublishE4Value(x));
    const vsOracle = diffArrays(ours, expected);
    expect(vsOracle.mismatches, describeMismatches('vs oracle', vsOracle)).toBe(0);
    const theirs = await run('publish e4 spelled out', inputs, (x) => nrDecodeE4m3(nrEncodeE4m3(nrF16Bits(x))));
    const vsTsl = diffArrays(ours, theirs);
    expect(vsTsl.mismatches, describeMismatches('vs the spelled-out TSL', vsTsl)).toBe(0);
  });
});
