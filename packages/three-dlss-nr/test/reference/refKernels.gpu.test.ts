// The per-kernel reference runner works in Node on our device, and the reference kernels agree with the CPU oracles.
//
// This is what the kernel chunks build on: if the reference port of OpenDLSS-NR (by maan, MIT) runs here, a TSL kernel
// can be compared with it byte for byte; where it cannot (see the device report), the CPU oracles in test/oracle/
// stand in, and these tests are what tie the oracles to the reference wherever the reference does run (CI).

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { alignUp } from '../../src/geometry.js';
import * as oracle from '../../src/numerics/oracle.js';
import { describeMismatches, diffArrays, SENTINEL_BYTE } from '../compare.js';
import { createGpuTestContext, type GpuTestContext } from '../gpu.js';
import { oracleGemmF16, oracleGemmFp8, type OracleGemmFp8Args } from '../oracle/gemm.js';
import { RefKernels } from './refKernels.js';

let gpu: GpuTestContext & { dispose(): void };
let ref: RefKernels;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  ref = await RefKernels.create(gpu.device);
});

afterAll(() => {
  ref?.destroy();
  gpu?.dispose();
});

const rng = new oracle.Xorshift(0x2545f491);

/** Random finite E4 codes (never the NaN codes), ~5% zeros. */
const randomE4 = (count: number): Uint8Array =>
  Uint8Array.from({ length: count }, () => {
    const draw = rng.next();
    if (draw % 20 === 0) return draw & 0x80;
    const code = (draw >>> 8) & 0xff;
    return (code & 0x7f) === 0x7f ? code & 0xfe : code;
  });

/** Random weight codes with |w| <= 9 (the bounded-half GEMM's load-time check, model.js:186-208), ~4% zeros. */
const randomWeights = (count: number): Uint8Array =>
  Uint8Array.from({ length: count }, () => {
    const draw = rng.next();
    if (draw % 25 === 0) return 0;
    return (draw & 0x80) | ((draw >>> 8) % 0x52);
  });

/** Random finite half patterns scaled into roughly [-4, 4] (no infinities or NaNs). */
const randomHalves = (count: number, maxExponent = 17): Uint16Array =>
  Uint16Array.from({ length: count }, () => {
    const draw = rng.next();
    const exponent = (draw >>> 16) % maxExponent;
    return (draw & 0x8000) | (exponent << 10) | (draw & 0x3ff);
  });

describe('reference runner on this device', () => {
  it('reports the device and which reference kernels it can run', () => {
    const info = gpu.adapter.info;
    const kernels = [
      'gemm_fp8',
      'window_attend',
      'gemm_f16',
      'vit_normalize',
      'vit_attend',
      'convert_f32_to_f16',
      'downsample',
      'upsample_residual',
      'post_blend',
    ];
    const lines = kernels.map((kernel) => `  ${kernel}: ${ref.unavailableReason(kernel) ?? 'available'}`);
    console.info(
      `[device] ${info.vendor} ${info.architecture} ${info.description}; shader-f16: ${gpu.shaderF16}; ` +
        `workgroup storage ${gpu.device.limits.maxComputeWorkgroupStorageSize}\n${lines.join('\n')}`,
    );
    expect(ref.f16).toBe(gpu.shaderF16);
    // The elementwise kernels have no f16 and no FXC-sensitive arithmetic: they run everywhere.
    expect(ref.unavailableReason('convert_f32_to_f16')).toBeUndefined();
  });

  it('runs ops.wgsl convert_f32_to_f16 and downsample, matching the oracle', async () => {
    const rows = 64;
    const features = new Float32Array(rows * 16);
    for (let i = 0; i < features.length; ++i) {
      // Exponents around the half range: subnormal, normal and overflowing halves.
      features[i] = oracle.f32FromBits(((rng.next() & 0x807fffff) | ((95 + (rng.next() % 54)) << 23)) >>> 0);
    }
    const inF32 = ref.tensor('features', rows, 16, 'f32', features);
    const outF16 = ref.tensor('features f16', rows, 16, 'f16');
    ref.fill(outF16, SENTINEL_BYTE);
    ref.op('convert_f32_to_f16', { count: rows * 16, channels: 16, inF32, outF16, label: 'features to half' });

    const [inWidth, inHeight, outWidth, outHeight, channels] = [16, 8, 8, 4, 32];
    const halves = randomHalves(inWidth * inHeight * channels);
    const inF16 = ref.tensor('pool in', inWidth * inHeight, channels, 'f16', halves);
    const outE4 = ref.tensor('pool out', outWidth * outHeight, channels, 'e4');
    ref.fill(outE4, SENTINEL_BYTE);
    ref.op('downsample', {
      count: outWidth * outHeight * channels,
      channels,
      inWidth,
      inHeight,
      outWidth,
      outHeight,
      inF16,
      outE4,
      label: 'pool',
    });
    await ref.run();

    const converted = new Uint16Array((await ref.read(outF16)).buffer);
    const expectedConverted = Array.from(features, (value) => oracle.f16Bits(value));
    const r1 = diffArrays(converted, expectedConverted);
    expect(r1.mismatches, describeMismatches('convert_f32_to_f16', r1, 4)).toBe(0);

    const pooled = await ref.read(outE4);
    const expectedPooled = new Uint8Array(outWidth * outHeight * channels);
    const half = (x: number, y: number, c: number) => oracle.f16ToNumber(halves[(y * inWidth + x) * channels + c]);
    for (let oy = 0; oy < outHeight; ++oy) {
      for (let ox = 0; ox < outWidth; ++ox) {
        for (let c = 0; c < channels; ++c) {
          const [sx, sy] = [ox * 2, oy * 2];
          const top = oracle.roundF16(half(sx, sy, c) + half(sx + 1, sy, c));
          const bottom = oracle.roundF16(half(sx, sy + 1, c) + half(sx + 1, sy + 1, c));
          const value = oracle.roundF16(oracle.roundF16(top + bottom) * 0.25);
          expectedPooled[(oy * outWidth + ox) * channels + c] = oracle.e4m3FromF16Bits(oracle.f16Bits(value));
        }
      }
    }
    const r2 = diffArrays(pooled, expectedPooled);
    expect(r2.mismatches, describeMismatches('downsample', r2, 2)).toBe(0);
  });
});

describe('reference gemm_f16 (shaders/gemm_f16.wgsl) vs the oracle', () => {
  it.for([
    { rows: 50, k: 16, n: 32, label: 'input adapter shape' },
    { rows: 64, k: 32, n: 4, label: 'head shape' },
  ])('$label: $rows x $k x $n, f32 + f16 + E4 outputs', async ({ rows, k, n, label }, context) => {
    const reason = ref.unavailableReason('gemm_f16');
    if (reason) return context.skip(reason);
    const paddedN = alignUp(n, 16);
    const input = randomHalves(rows * k, 16);
    const weights = randomHalves(k * paddedN, 15);
    const inputTensor = ref.tensor(`${label} in`, rows, k, 'f16', input);
    const outputs = {
      output: ref.tensor(`${label} e4`, rows, n, 'e4'),
      outputF16: ref.tensor(`${label} f16`, rows, n, 'f16'),
      outputF32: ref.tensor(`${label} f32`, rows, n, 'f32'),
    };
    for (const tensor of Object.values(outputs)) ref.fill(tensor, SENTINEL_BYTE);
    ref.gemmF16({ input: inputTensor, weights: ref.buffer(weights), paddedN, ...outputs, rows, k, n, label });
    await ref.run();
    const expected = oracleGemmF16({ rows, k, n, input, weights, paddedN });
    const f32 = new Uint32Array((await ref.read(outputs.outputF32)).buffer);
    const r1 = diffArrays(
      f32,
      Array.from(expected.f32, (v) => oracle.f32Bits(v)),
    );
    expect(r1.mismatches, describeMismatches('gemm_f16 f32', r1)).toBe(0);
    const r2 = diffArrays(new Uint16Array((await ref.read(outputs.outputF16)).buffer), expected.half);
    expect(r2.mismatches, describeMismatches('gemm_f16 f16', r2, 4)).toBe(0);
    const r3 = diffArrays(await ref.read(outputs.output), expected.e4);
    expect(r3.mismatches, describeMismatches('gemm_f16 e4', r3, 2)).toBe(0);
  });
});

describe('reference FP8 GEMM (composed src/matmul variants) vs the oracle', () => {
  const cases: (Omit<OracleGemmFp8Args, 'input' | 'weights' | 'residual' | 'scale'> & {
    label: string;
    residualFormat?: 'e4' | 'f16';
  })[] = [
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
    {
      label: 'partition 256 with E4 skip',
      rows: 32,
      k: 1024,
      n: 32,
      output: 'e4',
      partition: 256,
      residualFormat: 'e4',
    },
  ];

  it.for(cases)('$label', async (spec, context) => {
    const reason = ref.unavailableReason('gemm_fp8');
    if (reason) return context.skip(reason);
    const { rows, k, n, label, residualFormat } = spec;
    const batches = spec.batches ?? 1;
    const broadcast = spec.broadcast ?? false;
    const inputChannels = batches > 1 && !broadcast ? k * batches : k;
    const outputChannels = n * batches;
    const input = randomE4(rows * inputChannels);
    const weights = randomWeights(batches * k * n);
    const scale = residualFormat ? randomHalves(n, 15) : undefined;
    const residualData =
      residualFormat === 'e4'
        ? randomE4(rows * outputChannels)
        : residualFormat === 'f16'
          ? randomHalves(rows * outputChannels)
          : undefined;

    const inputTensor = ref.tensor(`${label} in`, rows, inputChannels, 'e4', input);
    const residual = residualFormat
      ? ref.tensor(`${label} skip`, rows, outputChannels, residualFormat, residualData)
      : null;
    const output = spec.output !== 'half' ? ref.tensor(`${label} e4`, rows, outputChannels, 'e4') : undefined;
    const outputF16 = spec.output !== 'e4' ? ref.tensor(`${label} f16`, rows, outputChannels, 'f16') : undefined;
    for (const tensor of [output, outputF16]) if (tensor) ref.fill(tensor, SENTINEL_BYTE);
    ref.gemm({
      input: inputTensor,
      weights: ref.fp8Matrix({ bytes: weights, k: k * batches, n, batchK: k, scales: scale }),
      output,
      outputF16,
      rows,
      k,
      n,
      batches,
      broadcast,
      partition: spec.partition ?? 0,
      silu: spec.silu ?? false,
      residual,
      label,
    });
    await ref.run();

    const expected = oracleGemmFp8({
      ...spec,
      input,
      inputChannels,
      weights,
      residual: residualFormat && residualData ? { format: residualFormat, data: residualData } : undefined,
      scale,
    });
    if (output) {
      const r = diffArrays(await ref.read(output), expected.e4!);
      expect(r.mismatches, describeMismatches(`${label} e4`, r, 2)).toBe(0);
    }
    if (outputF16) {
      const r = diffArrays(new Uint16Array((await ref.read(outputF16)).buffer), expected.half!);
      expect(r.mismatches, describeMismatches(`${label} f16`, r, 4)).toBe(0);
    }
  });
});
