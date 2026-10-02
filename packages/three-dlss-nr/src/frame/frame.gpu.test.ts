// The frame kernels (input_features, compose) against the reference's own frame.wgsl, run in Node on the same device
// through the reference's Kernels / Recorder (as its demo pipeline records them). Not bit-gated by design (exp, pow,
// log2, cos, sin, and backend-dependent fma contraction): the f32 arithmetic agrees to an ulp or so (1e-6 relative),
// which on the published half grid means a rare value one rounding step away (at most 2^-10 in code units, the
// Catmull-Rom history being a sum with negative weights); image bytes within one code value.
// The tests require that, and report how many values are not bit-identical (none on D3D12 so far; a handful of
// reprojected-history lanes on NVIDIA Vulkan).

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { DataTexture, HalfFloatType, RGBAFormat, RGFormat, FloatType } from 'three';
import { StorageTexture } from 'three/webgpu';
import { If, floatBitsToUint, ivec2, int, localId, textureLoad, workgroupId } from 'three/tsl';

import * as oracle from '../numerics/oracle.js';
import { attributeFromBytes, createTensor, readBuffer, wordAttribute, writeBuffer } from '../tensors.js';
import { kernel, kernelWGSL, runKernels } from '../tsl/KernelBuilder.js';
import { u } from '../tsl/packed.js';
import { fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { compareFeatures, halfUlps } from '../../test/oracle/frame.js';
import { RefKernels, referenceShader, type RefTensor } from '../../test/reference/refKernels.js';
import { expectIntegerComparisons } from '../../test/wgsl.js';
import { createCompose } from './compose.js';
import { NRFrameParams, type FrameColorSource, type FrameMotionSource, type NRFrameSettings } from './frameInputs.js';
import { createFrameKernels, NRHistory } from './history.js';
import { createInputFeatures } from './inputFeatures.js';

let gpu: GpuTestContext & { dispose(): void };
let ref: RefKernels;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  ref = await RefKernels.create(gpu.device);
  await ref.kernels.add(ref.numerics, referenceShader('frame.wgsl'), 'frame.wgsl', ['input_features', 'compose']);
});

afterAll(() => {
  ref?.destroy();
  gpu?.dispose();
});

const rng = new oracle.Xorshift(0x7a3d1c);
const uniform01 = () => rng.next() / 2 ** 32;

interface FrameData {
  /** rgba16float scene, validWidth * validHeight * 4 halves. */
  scene: Uint16Array;
  /** rg16float motion (uv, current -> previous, y down). */
  motion: Uint16Array;
  /** rgba16float history. */
  history: Uint16Array;
  /** f32 head [fullRows][4]. */
  head: Float32Array;
}

const GEOMETRY = { fullWidth: 48, fullHeight: 40, validWidth: 40, validHeight: 30 };

function randomFrame({ fullWidth, fullHeight, validWidth, validHeight } = GEOMETRY): FrameData {
  const pixels = validWidth * validHeight;
  const scene = new Uint16Array(pixels * 4);
  for (let i = 0; i < scene.length; ++i) {
    const draw = rng.next() % 200;
    // Linear HDR: mostly [0, 1.5], highlights to 20, a few negatives, infinities and NaNs.
    const value =
      draw === 0
        ? -uniform01()
        : draw === 1
          ? Infinity
          : draw === 2
            ? NaN
            : draw < 20
              ? uniform01() * 20
              : uniform01() * 1.5;
    scene[i] = i % 4 === 3 ? 0x3c00 : oracle.f16Bits(value);
  }
  const motion = new Uint16Array(pixels * 2);
  for (let p = 0; p < pixels; ++p) {
    // Up to +-3 pixels, some zero; edge pixels move off screen.
    const still = rng.next() % 5 === 0;
    motion[p * 2] = oracle.f16Bits(still ? 0 : ((uniform01() - 0.5) * 6) / validWidth);
    motion[p * 2 + 1] = oracle.f16Bits(still ? 0 : ((uniform01() - 0.5) * 6) / validHeight);
  }
  const history = new Uint16Array(pixels * 4);
  for (let i = 0; i < history.length; ++i) history[i] = i % 4 === 3 ? 0x3c00 : oracle.f16Bits(uniform01());
  const head = Float32Array.from({ length: fullWidth * fullHeight * 4 }, (_, i) =>
    i % 4 === 3 ? (uniform01() - 0.5) * 8 : (uniform01() - 0.5) * 2,
  );
  return { scene, motion, history, head };
}

/** The reference demo's 18-word parameter block (production-pipeline.js `params`). */
function referenceParams(settings: Required<NRFrameSettings>, flipY: boolean, imagePitch: number): Uint32Array {
  const words = new Uint32Array(18);
  const floats = new Float32Array(words.buffer);
  words.set([GEOMETRY.fullWidth, GEOMETRY.fullHeight, GEOMETRY.validWidth, GEOMETRY.validHeight]);
  words[4] = settings.seed;
  words[5] = settings.historyValid ? 1 : 0;
  words[6] = settings.enabled ? 1 : 0;
  words[7] = imagePitch;
  floats[8] = settings.paperWhite;
  floats[9] = settings.style;
  floats[10] = settings.localTone;
  floats[11] = settings.localStructure;
  floats[12] = settings.skinStructure;
  floats[13] = settings.autoMask ? 1 : 0;
  floats[14] = settings.blendScale;
  floats[15] = settings.intensity;
  floats[16] = settings.colorStrength;
  words[17] = flipY ? 1 : 0;
  return words;
}

const DEFAULTS: Required<NRFrameSettings> = {
  seed: 5,
  historyValid: true,
  enabled: true,
  paperWhite: 1,
  style: 0,
  localTone: 1,
  localStructure: 1,
  skinStructure: -1,
  autoMask: false,
  blendScale: 0.75,
  intensity: 1,
  colorStrength: 1,
};

interface ReferenceFrame {
  features: Float32Array;
  nextHistory: Uint16Array;
  image: Uint32Array;
}

/** One reference frame: input_features then compose, as the demo records them. */
async function referenceFrame(
  data: FrameData,
  settings: Required<NRFrameSettings>,
  flipY: boolean,
): Promise<ReferenceFrame> {
  const { fullWidth, fullHeight, validWidth, validHeight } = GEOMETRY;
  const pitch = validWidth;
  const params = referenceParams(settings, flipY, pitch);
  const scene = ref.buffer(data.scene, 'scene');
  const motion = ref.buffer(data.motion, 'motion');
  const history = ref.buffer(data.history, 'history');
  const head = ref.buffer(data.head, 'head');
  const features: RefTensor = ref.tensor('ref features', fullWidth * fullHeight, 16, 'f32');
  const next: RefTensor = ref.tensor('ref next history', validWidth * validHeight, 4, 'f16');
  const image: RefTensor = ref.tensor('ref image', validWidth * validHeight, 1, 'f32');
  for (const t of [features, next, image]) ref.fill(t, SENTINEL_BYTE);
  const recorder = ref.currentRecorder;
  recorder.pass(
    'input_features',
    { 1: scene, 2: history, 4: motion, 5: features.buffer },
    params,
    [Math.ceil(fullWidth / 8), Math.ceil(fullHeight / 8)],
    'input features',
  );
  recorder.pass(
    'compose',
    { 1: scene, 2: history, 3: head, 4: motion, 6: next.buffer, 7: image.buffer },
    params,
    [Math.ceil(validWidth / 8), Math.ceil(validHeight / 8)],
    'compose',
  );
  await ref.run();
  return {
    features: new Float32Array((await ref.read(features)).buffer),
    nextHistory: new Uint16Array((await ref.read(next)).buffer),
    image: new Uint32Array((await ref.read(image)).buffer),
  };
}

/** Our frame through buffer sources (the reference's layout). */
function ourBuffers(data: FrameData) {
  const { fullWidth, fullHeight, validWidth, validHeight } = GEOMETRY;
  const head = createTensor('head', fullWidth * fullHeight, 4, 'f32');
  writeBuffer(head, data.head);
  const features = createTensor('input features', fullWidth * fullHeight, 16, 'f32');
  const nextHistory = { attribute: wordAttribute(validWidth * validHeight * 8) };
  const image = { attribute: wordAttribute(validWidth * validHeight * 4) };
  fillSentinel(features);
  fillSentinel(nextHistory);
  fillSentinel(image);
  const scene = { attribute: attributeFromBytes(data.scene) };
  const motionBuffer = { attribute: attributeFromBytes(data.motion) };
  return {
    scene,
    motionBuffer,
    color: { buffer: scene } as FrameColorSource,
    motion: { buffer: motionBuffer } as FrameMotionSource,
    history: { attribute: attributeFromBytes(data.history) },
    head,
    features,
    nextHistory,
    image,
  };
}

interface Agreement {
  features: ReturnType<typeof compareFeatures>;
  historyMaxUlps: number;
  historyMismatches: number;
  imageMaxDelta: number;
  imageMismatches: number;
}

function agreement(
  features: Float32Array,
  nextHistory: Uint16Array,
  image: Uint32Array,
  expected: ReferenceFrame,
): Agreement {
  const { fullWidth, fullHeight } = GEOMETRY;
  let historyMaxUlps = 0;
  let historyMismatches = 0;
  for (let i = 0; i < expected.nextHistory.length; ++i) {
    if (nextHistory[i] === expected.nextHistory[i]) continue;
    historyMismatches += 1;
    historyMaxUlps = Math.max(
      historyMaxUlps,
      halfUlps(oracle.f16ToNumber(nextHistory[i]), oracle.f16ToNumber(expected.nextHistory[i])),
    );
  }
  let imageMaxDelta = 0;
  let imageMismatches = 0;
  for (let i = 0; i < expected.image.length; ++i) {
    if (image[i] === expected.image[i]) continue;
    imageMismatches += 1;
    for (let b = 0; b < 4; ++b) {
      const delta = Math.abs(((image[i] >>> (8 * b)) & 0xff) - ((expected.image[i] >>> (8 * b)) & 0xff));
      imageMaxDelta = Math.max(imageMaxDelta, delta);
    }
  }
  return {
    features: compareFeatures(features, expected.features, fullWidth * fullHeight),
    historyMaxUlps,
    historyMismatches,
    imageMaxDelta,
    imageMismatches,
  };
}

function expectAgreement(name: string, result: Agreement): void {
  const summary =
    `[frame ${name}] features: ${result.features.summary}; next history: ${result.historyMismatches} differ ` +
    `(max ${result.historyMaxUlps} half ulps); image: ${result.imageMismatches} pixels differ (max ${result.imageMaxDelta})`;
  console.info(summary);
  expect(result.features.maxAbsolute, summary).toBeLessThanOrEqual(2 ** -10);
  // Rare: fewer than 1 in 1000 published values may sit on the other side of a half-grid rounding.
  expect(result.features.mismatches, summary).toBeLessThan((GEOMETRY.fullWidth * GEOMETRY.fullHeight * 13) / 1000);
  expect(result.historyMismatches, summary).toBeLessThan((GEOMETRY.validWidth * GEOMETRY.validHeight * 4) / 1000);
  expect(result.features.noiseMaxHalfUlps, summary).toBeLessThanOrEqual(1);
  expect(result.historyMaxUlps, summary).toBeLessThanOrEqual(1);
  expect(result.imageMaxDelta, summary).toBeLessThanOrEqual(1);
}

const CASES: { name: string; settings: Partial<NRFrameSettings>; flipY?: boolean }[] = [
  { name: 'history valid, no style', settings: {} },
  { name: 'first frame (no history)', settings: { historyValid: false, seed: 77 } },
  { name: 'NR off', settings: { enabled: false } },
  {
    name: 'cinematic style, intensity, colour strength, auto mask, flipped rows',
    settings: {
      style: 1,
      intensity: 0.6,
      colorStrength: 0.5,
      autoMask: true,
      localTone: 0.8,
      localStructure: 0.4,
      paperWhite: 2,
    },
    flipY: true,
  },
  { name: 'natural style', settings: { style: 2, localTone: 1.5, skinStructure: 0.3, autoMask: true } },
];

describe('frame kernels vs the reference frame.wgsl', () => {
  it('one pair of kernels, uniforms updated between frames (R13)', async () => {
    const byFlip = new Map<
      boolean,
      {
        features: ReturnType<typeof createInputFeatures>;
        compose: ReturnType<typeof createCompose>;
        ours: ReturnType<typeof ourBuffers>;
        params: NRFrameParams;
      }
    >();
    for (const { name, settings, flipY = false } of CASES) {
      const data = randomFrame();
      const full = { ...DEFAULTS, ...settings };
      const expected = await referenceFrame(data, full, flipY);

      // Kernels are built once per flip (a baked constant) and reused across cases; only uniforms and data change.
      let built = byFlip.get(flipY);
      if (!built) {
        const ours = ourBuffers(data);
        const params = new NRFrameParams();
        const spec = { ...GEOMETRY, flipY };
        built = {
          ours,
          params,
          features: createInputFeatures(spec, { ...ours, params }),
          compose: createCompose(spec, { ...ours, params }),
        };
        byFlip.set(flipY, built);
      }
      const { ours, params } = built;
      writeBuffer(ours.scene, data.scene);
      writeBuffer(ours.motionBuffer, data.motion);
      writeBuffer(ours.history, data.history);
      writeBuffer(ours.head, data.head);
      for (const target of [ours.features, ours.nextHistory, ours.image]) fillSentinel(target);
      params.set(full);
      await runKernels(gpu.renderer, [built.features, built.compose]);

      const features = new Float32Array((await readBuffer(gpu.renderer, ours.features)).buffer);
      const nextHistory = new Uint16Array((await readBuffer(gpu.renderer, ours.nextHistory)).buffer);
      const image = new Uint32Array((await readBuffer(gpu.renderer, ours.image)).buffer);
      expectAgreement(name, agreement(features, nextHistory, image, expected));
    }
    const [k] = [...byFlip.values()];
    expectIntegerComparisons(kernelWGSL(gpu.renderer, k.features));
    expectIntegerComparisons(kernelWGSL(gpu.renderer, k.compose));
  });

  it('the per-pixel history flag keeps off-screen reprojections out of lanes 7-9 and the blend', async () => {
    const data = randomFrame();
    const ours = ourBuffers(data);
    const params = new NRFrameParams(DEFAULTS);
    const spec = { ...GEOMETRY, rejectOffscreenHistory: true };
    await runKernels(gpu.renderer, [createInputFeatures(spec, { ...ours, params })]);
    const features = new Float32Array((await readBuffer(gpu.renderer, ours.features)).buffer);
    const { fullWidth, validWidth, validHeight } = GEOMETRY;
    let rejected = 0;
    for (let y = 0; y < validHeight; ++y) {
      for (let x = 0; x < validWidth; ++x) {
        const mx = oracle.f16ToNumber(data.motion[(y * validWidth + x) * 2]);
        const my = oracle.f16ToNumber(data.motion[(y * validWidth + x) * 2 + 1]);
        const [ux, uy] = [Math.fround((x + 0.5) / validWidth) + mx, Math.fround((y + 0.5) / validHeight) + my];
        if (ux >= 0 && ux <= 1 && uy >= 0 && uy <= 1) continue;
        rejected += 1;
        const base = (y * fullWidth + x) * 16;
        expect([...features.subarray(base + 7, base + 10)]).toEqual([...features.subarray(base + 4, base + 7)]);
      }
    }
    expect(rejected).toBeGreaterThan(0);
  });
});

/** Read an RGBA texture into a buffer of f32 rgba (a tiny kernel; three has no storage-texture readback in compute). */
async function readTexture(texture: unknown, width: number, height: number): Promise<Float32Array> {
  const out = { attribute: wordAttribute(width * height * 16) };
  const k = kernel({
    label: 'read texture',
    kind: 'test_read_texture',
    workgroupSize: [8, 8],
    dispatch: [Math.ceil(width / 8), Math.ceil(height / 8)],
    inputs: {},
    outputs: { out },
    body: ({ out: target }) => {
      const x = workgroupId.x.mul(u(8)).add(localId.x).toVar();
      const y = workgroupId.y.mul(u(8)).add(localId.y).toVar();
      If(x.lessThan(u(width)).and(y.lessThan(u(height))), () => {
        const texel = textureLoad(texture, ivec2(int(x), int(y))).toVar();
        const base = y.mul(u(width)).add(x).mul(u(4)).toVar();
        for (const [c, value] of [texel.x, texel.y, texel.z, texel.w].entries()) {
          target.element(base.add(u(c))).assign(floatBitsToUint(value));
        }
      });
    },
  });
  await runKernels(gpu.renderer, [k]);
  return new Float32Array((await readBuffer(gpu.renderer, out, { validOnly: false })).buffer);
}

describe('texture sources and outputs (a WebGPURenderer render, no readback)', () => {
  it('colour and velocity textures give the same features and history as the buffer layout', async () => {
    const { validWidth, validHeight } = GEOMETRY;
    const data = randomFrame();
    // Finite scene values only, so the HDR output can be checked for finiteness.
    for (let i = 0; i < data.scene.length; ++i) if ((data.scene[i] & 0x7c00) === 0x7c00) data.scene[i] = 0x3c00;
    const params = new NRFrameParams(DEFAULTS);
    const viaBuffers = ourBuffers(data);
    const colorTexture = new DataTexture(data.scene, validWidth, validHeight, RGBAFormat, HalfFloatType);
    colorTexture.needsUpdate = true;
    // three's velocity: current NDC - previous NDC, y up = (-2 * mv.x, 2 * mv.y) of the uv motion (exact in f32).
    const velocity = new Float32Array(validWidth * validHeight * 2);
    for (let p = 0; p < validWidth * validHeight; ++p) {
      velocity[p * 2] = -2 * oracle.f16ToNumber(data.motion[p * 2]);
      velocity[p * 2 + 1] = 2 * oracle.f16ToNumber(data.motion[p * 2 + 1]);
    }
    const velocityTexture = new DataTexture(velocity, validWidth, validHeight, RGFormat, FloatType);
    velocityTexture.needsUpdate = true;
    const viaTextures = ourBuffers(data);
    const outputTexture = new StorageTexture(validWidth, validHeight);
    outputTexture.type = HalfFloatType;

    const textureSources = { color: { texture: colorTexture }, motion: { velocityTexture } };
    await runKernels(gpu.renderer, [
      createInputFeatures(GEOMETRY, { ...viaBuffers, params }),
      createCompose(GEOMETRY, { ...viaBuffers, params }),
      createInputFeatures(GEOMETRY, { ...viaTextures, ...textureSources, params }),
      createCompose(GEOMETRY, { ...viaTextures, ...textureSources, params, outputTexture }),
    ]);
    for (const key of ['features', 'nextHistory', 'image'] as const) {
      const a = await readBuffer(gpu.renderer, viaBuffers[key]);
      const b = await readBuffer(gpu.renderer, viaTextures[key]);
      expect(Buffer.compare(a, b), key).toBe(0);
    }
    // The HDR output is finite and, presented through the display transform, is the image (to one code value).
    const hdr = await readTexture(outputTexture, validWidth, validHeight);
    expect(hdr.every((v) => Number.isFinite(v))).toBe(true);
    expect(hdr.filter((_, i) => i % 4 === 3).every((a) => a === 1)).toBe(true);
  });

  it('with NR off, the HDR output texture is the rendered scene', async () => {
    const { validWidth, validHeight } = GEOMETRY;
    const data = randomFrame();
    for (let i = 0; i < data.scene.length; ++i) if ((data.scene[i] & 0x7c00) === 0x7c00) data.scene[i] = 0x3c00;
    const ours = ourBuffers(data);
    const outputTexture = new StorageTexture(validWidth, validHeight);
    outputTexture.type = HalfFloatType;
    const params = new NRFrameParams({ ...DEFAULTS, enabled: false });
    await runKernels(gpu.renderer, [createCompose(GEOMETRY, { ...ours, params, outputTexture })]);
    const hdr = await readTexture(outputTexture, validWidth, validHeight);
    for (let i = 0; i < data.scene.length; ++i) {
      const expected = i % 4 === 3 ? 1 : oracle.f16ToNumber(data.scene[i]);
      if (hdr[i] !== expected) expect(hdr[i], `component ${i}`).toBe(expected);
    }
  });
});

describe('NRHistory and createFrameKernels', () => {
  it('ping-pongs: compose[p] writes the buffer inputFeatures[p ^ 1] reads', async () => {
    const { fullWidth, fullHeight, validWidth, validHeight } = GEOMETRY;
    const data = randomFrame();
    const ours = ourBuffers(data);
    const history = new NRHistory(validWidth, validHeight);
    const params = new NRFrameParams({ ...DEFAULTS, historyValid: false, enabled: false });
    const frame = createFrameKernels(GEOMETRY, { ...ours, params, history });
    expect(frame.compose[0].label).toBe('compose');
    expect(frame.inputFeatures[1].label).toBe('input features');
    // Frame 0: NR off primes history 1 with the proxy; frame 1 (parity 1) with a still camera reads it back.
    await runKernels(gpu.renderer, [frame.inputFeatures[0], frame.compose[0]]);
    const primed = new Uint16Array((await readBuffer(gpu.renderer, history.write(0), { validOnly: false })).buffer);
    expect(primed.some((h) => h !== 0)).toBe(true);
    params.set({ historyValid: true, enabled: true });
    const still = { ...ours, motion: null, params, history };
    const frame1 = createFrameKernels(GEOMETRY, still);
    await runKernels(gpu.renderer, [frame1.inputFeatures[1]]);
    const features = new Float32Array((await readBuffer(gpu.renderer, ours.features)).buffer);
    // With zero motion the Catmull-Rom at a pixel centre is that pixel: lanes 7-9 = centre(truncated proxy).
    for (let y = 0; y < validHeight; ++y) {
      for (let x = 0; x < validWidth; ++x) {
        const base = (y * fullWidth + x) * 16;
        for (let c = 0; c < 3; ++c) {
          const stored = oracle.f16ToNumber(primed[(y * validWidth + x) * 4 + c]);
          const centred = oracle.roundF16(oracle.roundF16(oracle.roundF16(stored) - 0.5) * 0.125);
          expect(halfUlps(features[base + 7 + c], centred), `pixel ${x},${y} lane ${7 + c}`).toBeLessThanOrEqual(1);
        }
      }
    }
    expect(fullHeight).toBeGreaterThan(validHeight);
  });
});
