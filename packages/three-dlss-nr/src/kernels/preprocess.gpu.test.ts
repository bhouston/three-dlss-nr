// preprocess (port of preprocess.wgsl) against the reference's own preprocess kernel - the dispatch
// `Network.featuresFromProxy` records - run in Node on the same device. Lanes 3-15 must be bit-identical; the three
// noise lanes go through the hardware's log2 / cos / sin and are only expected to agree to the half-grid rounding
// (they are reported, and bit-identical on the devices tried so far).

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import * as oracle from '../numerics/oracle.js';
import { attributeFromBytes, createTensor, readBuffer } from '../tensors.js';
import { kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import { fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { RefKernels, referenceShader } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons } from '../../test/wgsl.js';
import { createPreprocess, type PreprocessSpec } from './preprocess.js';
import { compareFeatures } from '../../test/oracle/frame.js';

let gpu: GpuTestContext & { dispose(): void };
let ref: RefKernels;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  ref = await RefKernels.create(gpu.device);
  await ref.kernels.add(ref.numerics, referenceShader('preprocess.wgsl'), 'preprocess.wgsl', ['preprocess']);
});

afterAll(() => {
  ref?.destroy();
  gpu?.dispose();
});

const rng = new oracle.Xorshift(0x9e3779b9);

/** Code values mostly in [0, 1], some outside, some exactly on half midpoints. */
const randomProxy = (count: number): Float32Array =>
  Float32Array.from({ length: count }, () => {
    const draw = rng.next();
    if (draw % 29 === 0) return oracle.f16ToNumber(draw & 0x3bff) + 2 ** -12; // between two halves
    return ((draw >>> 8) / 2 ** 24) * 1.2 - 0.1;
  });

/** network.js `featuresFromProxy`'s parameter block. */
function referenceParams(spec: PreprocessSpec): Uint32Array {
  const words = new Uint32Array(12);
  const floats = new Float32Array(words.buffer);
  words.set([spec.fullWidth, spec.fullHeight, spec.validWidth, spec.validHeight, spec.sourceWidth, spec.sourceHeight]);
  words[6] = spec.seed;
  floats[7] = spec.autoMask ? 1 : -1;
  floats[8] = spec.localTone ?? 1;
  floats[9] = spec.localStructure ?? 1;
  floats[10] = spec.skinStructure ?? -1;
  floats[11] = spec.style ?? 0;
  return words;
}

describe('preprocess vs the reference preprocess.wgsl', () => {
  it.for<PreprocessSpec & { name: string }>([
    {
      name: 'proxy at valid size',
      fullWidth: 48,
      fullHeight: 40,
      validWidth: 40,
      validHeight: 30,
      sourceWidth: 40,
      sourceHeight: 30,
      seed: 7,
    },
    {
      name: 'larger proxy, auto mask, style',
      fullWidth: 72,
      fullHeight: 64,
      validWidth: 61,
      validHeight: 45,
      sourceWidth: 128,
      sourceHeight: 90,
      seed: 123456,
      autoMask: true,
      localTone: 0.7,
      localStructure: 0.3,
      skinStructure: -1,
      style: 2,
    },
  ])('$name', async (spec) => {
    const proxy = randomProxy(spec.sourceWidth * spec.sourceHeight * 4);
    const rows = spec.fullWidth * spec.fullHeight;

    const refFeatures = ref.tensor(`features ${spec.name}`, rows, 16, 'f32');
    ref.fill(refFeatures, SENTINEL_BYTE);
    ref.currentRecorder.pass(
      'preprocess',
      { 1: ref.buffer(proxy), 5: refFeatures.buffer },
      referenceParams(spec),
      [Math.ceil(spec.fullWidth / 8), Math.ceil(spec.fullHeight / 8)],
      'preprocess',
    );
    await ref.run();
    const expected = new Float32Array((await ref.read(refFeatures, { validOnly: false })).buffer);

    const features = createTensor('features', rows, 16, 'f32');
    fillSentinel(features);
    const k = createPreprocess(spec, { proxy: { attribute: attributeFromBytes(proxy) }, features });
    await runKernels(gpu.renderer, [k]);
    const actual = new Float32Array((await readBuffer(gpu.renderer, features, { validOnly: false })).buffer);

    const report = compareFeatures(actual, expected, rows);
    console.info(`[preprocess ${spec.name}] ${report.summary}`);
    expect(report.exactLanes, report.summary).toEqual(Array.from({ length: 13 }, () => true));
    expect(report.noiseMaxHalfUlps, report.summary).toBeLessThanOrEqual(1);
    // The padding rows of the allocation stay untouched on both sides.
    expect(actual.subarray(rows * 16)).toEqual(expected.subarray(rows * 16));
    expectIntegerComparisons(kernelWGSL(gpu.renderer, k));
  });
});
