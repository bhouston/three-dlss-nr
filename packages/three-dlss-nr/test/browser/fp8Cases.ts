// The seven FP8 GEMM reference-check cases of test/reference/refKernels.gpu.test.ts, with byte-identical inputs.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). refKernels.gpu.test.ts draws every input from
// one module-level xorshift stream, in test order, so the inputs a CI run used depend on the draws of the tests before
// it. `fp8ReferenceCases()` replays that stream exactly (ops test, gemm_f16 tests, then the FP8 cases), so a result
// computed here (in a browser, by the oracle, ...) can be compared with what a CI log reports for the same index.
// Keep it in step with refKernels.gpu.test.ts.

import * as oracle from '../../src/numerics/oracle.js';
import type { OracleGemmFp8Args } from '../oracle/gemm.js';

export interface Fp8Case {
  label: string;
  rows: number;
  k: number;
  n: number;
  batches: number;
  broadcast: boolean;
  partition: 0 | 256 | 512 | 1024;
  silu: boolean;
  output: 'e4' | 'half' | 'dual';
  residualFormat?: 'e4' | 'f16';
  inputChannels: number;
  outputChannels: number;
  input: Uint8Array;
  weights: Uint8Array;
  scale?: Uint16Array;
  residualData?: Uint8Array | Uint16Array;
}

type CaseSpec = Omit<OracleGemmFp8Args, 'input' | 'weights' | 'residual' | 'scale'> & {
  label: string;
  residualFormat?: 'e4' | 'f16';
};

/** The case list of refKernels.gpu.test.ts ('reference FP8 GEMM ... vs the oracle'), in order. */
export const FP8_CASE_SPECS: CaseSpec[] = [
  { label: 'plain 32x32x32 e4', rows: 32, k: 32, n: 32, output: 'e4' },
  { label: 'rows 50, K 64, SiLU', rows: 50, k: 64, n: 32, output: 'e4', silu: true },
  { label: 'qkv half output', rows: 32, k: 32, n: 96, output: 'half' },
  { label: 'dual output, E4 skip', rows: 32, k: 128, n: 32, output: 'dual', residualFormat: 'e4' },
  { label: 'E4 output, f16 skip', rows: 32, k: 32, n: 32, output: 'e4', residualFormat: 'f16' },
  {
    label: 'expert expand (batched, broadcast, SiLU)',
    rows: 32,
    k: 64,
    n: 128,
    batches: 2,
    broadcast: true,
    output: 'e4',
    silu: true,
  },
  { label: 'partition 256 with E4 skip', rows: 32, k: 1024, n: 32, output: 'e4', partition: 256, residualFormat: 'e4' },
];

/** The FP8 cases with the inputs refKernels.gpu.test.ts generated for them (gemm_f16 tests not skipped). */
export function fp8ReferenceCases(): Fp8Case[] {
  const rng = new oracle.Xorshift(0x2545f491);
  const randomE4 = (count: number): Uint8Array =>
    Uint8Array.from({ length: count }, () => {
      const draw = rng.next();
      if (draw % 20 === 0) return draw & 0x80;
      const code = (draw >>> 8) & 0xff;
      return (code & 0x7f) === 0x7f ? code & 0xfe : code;
    });
  const randomWeights = (count: number): Uint8Array =>
    Uint8Array.from({ length: count }, () => {
      const draw = rng.next();
      if (draw % 25 === 0) return 0;
      return (draw & 0x80) | ((draw >>> 8) % 0x52);
    });
  const randomHalves = (count: number, maxExponent = 17): Uint16Array =>
    Uint16Array.from({ length: count }, () => {
      const draw = rng.next();
      const exponent = (draw >>> 16) % maxExponent;
      return (draw & 0x8000) | (exponent << 10) | (draw & 0x3ff);
    });

  // 'runs ops.wgsl convert_f32_to_f16 and downsample': 64 * 16 features of two draws each, then 16 * 8 * 32 halves.
  for (let i = 0; i < 64 * 16 * 2; ++i) rng.next();
  randomHalves(16 * 8 * 32);
  // gemm_f16: (50 x 16 -> 32) and (64 x 32 -> 4), inputs then weights [k][alignUp(n, 16)].
  randomHalves(50 * 16, 16);
  randomHalves(16 * 32, 15);
  randomHalves(64 * 32, 16);
  randomHalves(32 * 16, 15);

  return FP8_CASE_SPECS.map((spec) => {
    const batches = spec.batches ?? 1;
    const broadcast = spec.broadcast ?? false;
    const inputChannels = batches > 1 && !broadcast ? spec.k * batches : spec.k;
    const outputChannels = spec.n * batches;
    const input = randomE4(spec.rows * inputChannels);
    const weights = randomWeights(batches * spec.k * spec.n);
    const scale = spec.residualFormat ? randomHalves(spec.n, 15) : undefined;
    const residualData =
      spec.residualFormat === 'e4'
        ? randomE4(spec.rows * outputChannels)
        : spec.residualFormat === 'f16'
          ? randomHalves(spec.rows * outputChannels)
          : undefined;
    return {
      label: spec.label,
      rows: spec.rows,
      k: spec.k,
      n: spec.n,
      batches,
      broadcast,
      partition: spec.partition ?? 0,
      silu: spec.silu ?? false,
      output: spec.output,
      residualFormat: spec.residualFormat,
      inputChannels,
      outputChannels,
      input,
      weights,
      scale,
      residualData,
    };
  });
}
