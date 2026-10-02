// Browser page of the parity suite: one three.js frame per scene, the same input features into three renderers, every
// tensor compared byte for byte, deterministic visualizations, and per-scene timings.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Bundled and driven by scripts/generate.mjs in a
// real Chrome: the reference needs `shader-f16`, which Dawn in Node lacks on the Windows dev machine and which
// lavapipe's software f16 folds away. The three renderers:
//   opendlss-nr        the reference WebGPU port, standalone: upstream `Network.create` on its own device, its WGSL
//                      fetched from the submodule, its own weight loader (the comparison's reference);
//   three-dlss-nr-tsl  the native TSL port (`tslBackend`, `NRNetwork`) on a three.js WebGPURenderer;
//   three-dlss-nr-shim the reference WGSL vendored into the library (`referenceWgslBackend`) on the renderer's device.
// Results are POSTed to the generator (`uploadBase`), which writes the PNGs and JSON.

import { Network } from '@ref/network.js';
import { composeImage } from '@ref/parity.js';
import { HalfFloatType, RenderTarget, WebGPURenderer } from 'three/webgpu';
import {
  createNRDevice,
  geometryFromValid,
  NRFrameTimer,
  NRModel,
  readBuffer,
  summarizeMilliseconds,
  tslBackend,
  type NRBackend,
  type NRFrameTiming,
  type NRGeometry,
} from 'three-dlss-nr';
import { loadReferenceModel, referenceWgslBackend } from 'three-dlss-nr/reference-backend';

import { bytesOf, compareBytes, type TensorVerdict } from './compare.js';
import { featuresFromFrame } from './features.js';
import { SCENES } from './scenes.js';
import {
  blendWeight,
  boundaryCodes,
  fitForDisplay,
  fromRgba,
  headRgb,
  inputProxy,
  type RgbImage,
} from './visualize.js';

const log = (...args: unknown[]) => console.log(...args);

export const REFERENCE_ID = 'opendlss-nr';
export const TSL_ID = 'three-dlss-nr-tsl';
export const SHIM_ID = 'three-dlss-nr-shim';

/** Visualized boundaries: name, level (-1 = the full field) and channels. */
export const VISUAL_BOUNDARIES = [
  { output: 'block-0', name: 'block-0', level: -1, channels: 32 },
  { output: 'block-14', name: 'block-14', level: 2, channels: 128 },
  { output: 'vit-38', name: 'block-38', level: 5, channels: 1024 },
  { output: 'block-69', name: 'block-69', level: 0, channels: 32 },
] as const;

const POST_TENSORS = ['post merge', 'post merge raw', 'post block raw'];

interface Options {
  modelUrl: string;
  shaderBase: string;
  modelsBase: string;
  uploadBase: string;
  sizes: [number, number][];
  scenes?: string[];
  warmup: number;
  frames: number;
}

/** What every renderer offers the page (the standalone reference is adapted to it). */
interface Renderer {
  id: string;
  dispatches: number;
  writeFeatures(features: Float32Array): void;
  run(timing: boolean): Promise<NRFrameTiming>;
  readFeatures(): Promise<Uint8Array>;
  readHead(): Promise<Float32Array>;
  readBoundary(name: string): Promise<Uint8Array>;
  readTensor(label: string): Promise<Uint8Array>;
  boundaryNames: readonly string[];
  dispose(): void;
}

async function upload(base: string, path: string, body: BodyInit): Promise<void> {
  const response = await fetch(`${base}${path}`, { method: 'POST', body });
  if (!response.ok) throw new Error(`upload ${path}: ${response.status} ${await response.text()}`);
}

const uploadImage = (base: string, path: string, image: RgbImage) =>
  upload(base, `${path}?width=${image.width}&height=${image.height}`, image.data as unknown as BodyInit);

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as unknown as BufferSource));
  return [...digest].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function backendRenderer(id: string, backend: NRBackend, renderer: any): Renderer {
  return {
    id,
    dispatches: backend.dispatchCount,
    writeFeatures: (features) => backend.writeFeatures(features),
    run: (timing) => backend.run({ timing }),
    readFeatures: () => readBuffer(renderer, backend.features),
    readHead: () => backend.readHead(),
    readBoundary: (name) => backend.readBoundary(name),
    readTensor: (label) => backend.readTensor(label),
    boundaryNames: backend.boundaryNames,
    dispose: () => backend.dispose(),
  };
}

/** The upstream port as is, on its own device; timed with the same queue brackets as the backends. */
async function standaloneRenderer(options: Options, width: number, height: number, capture: boolean) {
  const network = await Network.create({
    weights: options.modelUrl.replace(/\/+$/, ''),
    width,
    height,
    captureBoundaries: capture,
    shaderBase: new URL(options.shaderBase, location.href),
  });
  const timer = new NRFrameTimer(network.device);
  const renderer: Renderer = {
    id: REFERENCE_ID,
    dispatches: network.recorder.dispatchCount,
    writeFeatures: (features) => network.writeFeatures(features),
    async run(timing) {
      if (!network.validated) return timer.measure(() => network.run(), { gpu: false });
      return timer.measure(
        () => {
          const encoder = network.device.createCommandEncoder({ label: 'nr frame' });
          network.recorder.encode(encoder);
          network.device.queue.submit([encoder.finish()]);
        },
        { gpu: timing },
      );
    },
    readFeatures: async () => (await network.readTensorByLabel('input features')).bytes,
    readHead: () => network.readHead(),
    readBoundary: (name) => network.readBoundary(name),
    readTensor: async (label) => (await network.readTensorByLabel(label)).bytes,
    boundaryNames: network.boundaryNames,
    dispose: () => {
      timer.dispose();
      network.destroy();
    },
  };
  return { renderer, blendScale: network.model.blendScale() as number, adapter: network.adapterInfo ?? {} };
}

/** Render one scene into an RGBA16F target and read the linear colour back (rows top to bottom). */
async function renderScene(renderer: any, sceneId: string, options: Options, width: number, height: number) {
  const entry = SCENES.find((s) => s.id === sceneId)!;
  const { scene, camera, dispose } = await entry.create(renderer, options.modelsBase, width / height);
  const target = new RenderTarget(width, height, { type: HalfFloatType, depthBuffer: true, samples: 0 });
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
  target.dispose();
  dispose();
  const halves = new Uint16Array(pixels.buffer, pixels.byteOffset, width * height * 4);
  const linear = new Float32Array(width * height * 4);
  for (let i = 0; i < linear.length; ++i) linear[i] = halfToNumber(halves[i]);
  return linear;
}

function halfToNumber(bits: number): number {
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  const sign = bits & 0x8000 ? -1 : 1;
  if (exponent === 0x1f) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24);
}

/** Every parity tensor of one renderer: the features it ran on, all boundaries, the post tensors, the head. */
async function readTensors(renderer: Renderer): Promise<Map<string, Uint8Array>> {
  const tensors = new Map<string, Uint8Array>();
  tensors.set('input features', await renderer.readFeatures());
  for (const name of renderer.boundaryNames) tensors.set(name, await renderer.readBoundary(name));
  for (const label of POST_TENSORS) tensors.set(label, await renderer.readTensor(label));
  tensors.set('head', bytesOf(await renderer.readHead()));
  return tensors;
}

const asF32 = (bytes: Uint8Array) => new Float32Array(bytes.slice().buffer);

function visualize(
  tensors: Map<string, Uint8Array>,
  geometry: NRGeometry,
  blendScale: number,
): Record<string, RgbImage> {
  const features = asF32(tensors.get('input features')!);
  const head = asF32(tensors.get('head')!);
  const composed = composeImage(head, { features }, geometry);
  const images: Record<string, RgbImage> = {
    input: inputProxy(features, geometry),
    output: fromRgba(composed.data, composed.width, composed.height),
    'head-rgb': headRgb(head, geometry),
    'blend-logit': blendWeight(head, geometry, blendScale),
  };
  for (const { output, name, level, channels } of VISUAL_BOUNDARIES) {
    const size = level < 0 ? { width: geometry.fullWidth, height: geometry.fullHeight } : geometry.levels[level];
    images[output] = boundaryCodes(tensors.get(name)!, size.width, size.height, channels);
  }
  for (const key of Object.keys(images)) images[key] = fitForDisplay(images[key]);
  return images;
}

async function createRenderers(
  options: Options,
  renderer: any,
  models: { nr: NRModel; reference: any },
  width: number,
  height: number,
  capture: boolean,
) {
  log(`${width}x${height}: creating the standalone reference${capture ? ' (capturing boundaries)' : ''}`);
  const standalone = await standaloneRenderer(options, width, height, capture);
  log(`${width}x${height}: creating the shim`);
  const shim = await referenceWgslBackend.create({
    renderer,
    model: models.reference,
    width,
    height,
    captureBoundaries: capture,
  });
  log(`${width}x${height}: creating the TSL network (compiling)`);
  let started = performance.now();
  const tsl = await tslBackend.create({
    renderer,
    model: models.nr as any,
    width,
    height,
    captureBoundaries: capture,
    onProgress: (message: string) => {
      if (performance.now() - started > 5000) {
        started = performance.now();
        log(`  ${message}`);
      }
    },
  });
  return {
    standalone,
    all: [standalone.renderer, backendRenderer(TSL_ID, tsl, renderer), backendRenderer(SHIM_ID, shim, renderer)],
  };
}

async function generate(options: Options) {
  const { device } = await createNRDevice();
  if (!device.features.has('shader-f16'))
    throw new Error('this browser/device has no shader-f16; the reference needs it');
  const renderer = new WebGPURenderer({ device, antialias: false });
  await renderer.init();
  const info = (device as any).adapterInfo ?? {};
  const adapter = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim();
  log(`device: ${adapter}`);
  const scenes = SCENES.filter((s) => !options.scenes || options.scenes.includes(s.id));
  const models = {
    nr: await NRModel.load(options.modelUrl),
    reference: await loadReferenceModel(renderer, options.modelUrl),
  };
  const summary: unknown[] = [];

  for (const [width, height] of options.sizes) {
    const geometry = geometryFromValid(width, height);
    const sizeId = `${width}x${height}`;
    const inputs = new Map<string, Float32Array>();
    for (const scene of scenes) {
      log(`${sizeId}: rendering ${scene.id}`);
      inputs.set(scene.id, featuresFromFrame(await renderScene(renderer, scene.id, options, width, height), geometry));
    }

    // Parity: every renderer gets the same features; every tensor is compared with the reference.
    const parity = await createRenderers(options, renderer, models, width, height, true);
    const [reference, ...others] = parity.all;
    for (const scene of scenes) {
      const features = inputs.get(scene.id)!;
      const tensors = new Map<string, Map<string, Uint8Array>>();
      for (const r of parity.all) {
        r.writeFeatures(features);
        await r.run(false);
        tensors.set(r.id, await readTensors(r));
      }
      const expected = tensors.get(reference.id)!;
      const verdicts: Record<string, Record<string, TensorVerdict>> = {};
      for (const r of others) {
        const actual = tensors.get(r.id)!;
        if ([...actual.keys()].join() !== [...expected.keys()].join()) throw new Error(`${r.id}: tensor names differ`);
        verdicts[r.id] = Object.fromEntries(
          [...expected].map(([name, bytes]) => [name, compareBytes(actual.get(name)!, bytes)]),
        );
      }
      const digests: Record<string, string> = {};
      for (const [name, bytes] of expected) digests[name] = await sha256(bytes);
      const path = `${sizeId}/${scene.id}`;
      for (const r of parity.all) {
        const images = visualize(tensors.get(r.id)!, geometry, parity.standalone.blendScale);
        for (const [output, image] of Object.entries(images))
          await uploadImage(options.uploadBase, `${path}/${output}/${r.id}`, image);
      }
      const failures = Object.values(verdicts)
        .flatMap((v) => Object.values(v))
        .filter((v) => v.verdict !== 'bit-exact');
      log(`${path}: ${expected.size} tensors x ${others.length} renderers, ${failures.length} not bit-exact`);
      await upload(
        options.uploadBase,
        `${path}/parity.json`,
        JSON.stringify({
          scene: scene.id,
          title: `${scene.title}, ${sizeId}`,
          description: scene.description,
          tags: [...scene.tags, sizeId],
          size: sizeId,
          adapter,
          geometry: {
            valid: sizeId,
            field: `${geometry.fullWidth}x${geometry.fullHeight}`,
            levels: geometry.levels.map((l) => `${l.width}x${l.height}`),
            vitTokens: geometry.vitTokens,
          },
          dispatches: Object.fromEntries(parity.all.map((r) => [r.id, r.dispatches])),
          blendScale: parity.standalone.blendScale,
          tensors: [...expected].map(([name, bytes]) => ({ name, bytes: bytes.length, sha256: digests[name] })),
          verdicts,
        }),
      );
      summary.push({ scene: path, failures: failures.length });
    }
    for (const r of parity.all) r.dispose();

    // Timing: the same three renderers without boundary captures, interleaved frame by frame.
    const timed = await createRenderers(options, renderer, models, width, height, false);
    for (const scene of scenes) {
      const features = inputs.get(scene.id)!;
      const samples = new Map<string, NRFrameTiming[]>(timed.all.map((r) => [r.id, []]));
      for (const r of timed.all) {
        r.writeFeatures(features);
        await r.run(false);
      }
      for (let i = 0; i < options.warmup + options.frames; ++i) {
        for (const r of timed.all) {
          const timing = await r.run(true);
          if (i >= options.warmup) samples.get(r.id)!.push(timing);
        }
      }
      const rows = Object.fromEntries(
        timed.all.map((r) => {
          const list = samples.get(r.id)!;
          return [
            r.id,
            {
              method: list[0].method,
              dispatches: r.dispatches,
              frames: list.length,
              gpu: summarizeMilliseconds(list.map((t) => t.gpuMilliseconds)),
              wall: summarizeMilliseconds(list.map((t) => t.wallMilliseconds)),
            },
          ];
        }),
      );
      log(`${sizeId}/${scene.id} timing: ${JSON.stringify(rows)}`);
      await upload(
        options.uploadBase,
        `${sizeId}/${scene.id}/timing.json`,
        JSON.stringify({ adapter, warmup: options.warmup, renderers: rows }),
      );
    }
    for (const r of timed.all) r.dispose();
  }
  models.nr.dispose();
  models.reference.dispose();
  return { adapter, summary };
}

/** Reject as soon as any WebGPU device is lost (e.g. a TDR on a busy GPU) instead of hanging the harness. */
function failOnDeviceLoss<A extends unknown[], R>(run: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
  return (...args: A) =>
    new Promise<R>((resolve, reject) => {
      const original = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
        const device = await original.call(this, descriptor);
        void device.lost.then((info) => {
          if (info.reason !== 'destroyed') reject(new Error(`WebGPU device lost: ${info.message}`));
        });
        return device;
      };
      run(...args).then(resolve, reject);
    });
}

(globalThis as any).nrFidelity = { generate: failOnDeviceLoss(generate) };
