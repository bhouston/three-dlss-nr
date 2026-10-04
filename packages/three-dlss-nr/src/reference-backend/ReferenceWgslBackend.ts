// The "reference-wgsl" NR backend: OpenDLSS-NR's own WebGPU port, run unchanged on three.js' GPUDevice.
//
// THIS BACKEND IS THE UPSTREAM CODE. Every kernel, pipeline, buffer layout and dispatch it runs comes from the
// reference WebGPU port of OpenDLSS-NR by maan (https://github.com/maanHimself/OpenDLSS-NR,
// ports/browser-webgpu, MIT, Copyright (c) 2026 maan), pinned at commit 9d08f41, vendored byte for byte into
// vendor/opendlss-nr/ at build time (scripts/bundle-reference.mjs). This file only does what the upstream
// `Network.create` does (src/network.js), with four differences:
//   * the WGSL comes from the vendored string constants instead of fetch() (a package cannot rely on a server
//     layout), and the frame shader can be added the way the upstream demo adds it (`extraShaders`);
//   * the device is `renderer.backend.device`, and the input features and the head are three.js storage
//     attributes whose GPU buffers the upstream graph writes and reads directly (GPU-resident I/O, no copies);
//   * a model given in memory is uploaded with the same steps as the upstream `Model.load` (which only fetches).
//   * shader modules and layouts are kept on the device across rebuilds, so the upstream pipeline cache can hit.
// It implements the backend-neutral `NRBackend` interface, so it can be swapped with the native TSL port.

import {
  geometryFromValid,
  Graph,
  Kernels,
  Matmul,
  Model,
  Network,
  readBack,
  Recorder,
  requireWorkgroupStorage,
  SHADERS,
  SOURCE,
  Tensors,
  WINDOW_WORKGROUP_STORAGE,
  WindowAttention,
} from '../../vendor/opendlss-nr/index.js';
import {
  rendererDevice,
  unmetRequirements,
  type NRBackend,
  type NRBackendCreateOptions,
  type NRBackendFactory,
  type NRBackendGeometry,
  type NRBackendMemory,
  type NRBackendModelSource,
  type NRBackendRequirements,
  type NRFrameTiming,
  type NRManifestLike,
  type NRRunOptions,
} from '../backend/NRBackend.js';
import { BUFFER_USAGE } from '../backend/gpuFlags.js';
import { NRFrameTimer } from '../backend/timing.js';
import { createTensor } from '../tensors.js';
import type { NRTensor } from '../types.js';

/** The upstream source the vendored code was taken from (repository, commit, SHA-256 per file). */
export const REFERENCE_SOURCE: { readonly repository: string; readonly commit: string } = SOURCE;

/**
 * What the reference needs: `shader-f16` (its FP8 GEMM, window attention and SiLU-table modules begin
 * `enable f16;`), 32 KiB of workgroup storage (window attention, `window/index.js`), and its eight storage bindings.
 */
export const REFERENCE_REQUIREMENTS: NRBackendRequirements = {
  features: ['shader-f16'],
  limits: {
    maxComputeWorkgroupStorageSize: WINDOW_WORKGROUP_STORAGE,
    maxStorageBuffersPerShaderStage: 8,
    maxComputeInvocationsPerWorkgroup: 512,
  },
};

const HOW_TO_GET_F16 =
  'Create the renderer on a device that has it: `new WebGPURenderer({ device: (await createNRDevice()).device })` ' +
  "(createNRDevice requests 'shader-f16' and the raised limits when the adapter offers them), or let three create " +
  'the device - WebGPURenderer requests every feature the adapter supports, but only default limits, so also pass ' +
  '`requiredLimits: { maxComputeWorkgroupStorageSize: 32768, maxStorageBufferBindingSize: adapter.limits.' +
  'maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }`. If the adapter itself lacks ' +
  "'shader-f16' (e.g. Dawn in Node on Windows, or a GPU without half support) only the 'tsl' backend can run.";

/** Why the reference cannot run on `device`, or null. */
export function referenceUnavailableReason(device: GPUDevice): string | null {
  const problems = unmetRequirements(device, REFERENCE_REQUIREMENTS);
  if (!problems.length) return null;
  return `the reference-wgsl backend cannot run on this device: ${problems.join('; ')}. ${HOW_TO_GET_F16}`;
}

// -------------------------------------------------------------------------------------------------------------------
// Weights
// -------------------------------------------------------------------------------------------------------------------

/** A reference `Model` (vendored `src/model.js`) with its GPU buffers; share one across backends and resizes. */
export interface ReferenceModel {
  readonly isReferenceModel: true;
  readonly device: GPUDevice;
  /** The upstream `Model` instance. */
  readonly model: any;
  /** Bytes uploaded so far (stages plus the re-laid-out buffers the graphs built). */
  readonly bytesUploaded: number;
  dispose(): void;
}

const isReferenceModel = (value: unknown): value is ReferenceModel =>
  typeof value === 'object' && value !== null && (value as ReferenceModel).isReferenceModel === true;

/** The upstream `Model.load`, minus the fetches: the same checks, the same 4 bytes of slack per stage. */
function modelFromStages(device: GPUDevice, manifest: NRManifestLike, stages: ReadonlyMap<string, Uint8Array>): any {
  const model = new Model(device);
  model.blockCount = manifest.totals.blockCount;
  for (const stage of manifest.stages) {
    const bytes = stages.get(stage.id);
    if (!bytes) throw new Error(`cannot read stage ${stage.id}`);
    if (bytes.byteLength !== stage.packedByteLength) throw new Error(`stage size mismatch: ${stage.id}`);
  }
  for (const entry of manifest.tensors) {
    const stage = stages.get(entry.stage);
    if (!stage) throw new Error(`tensor references unknown stage ${entry.stage}`);
    if (entry.stageOffset + entry.byteLength > stage.byteLength) throw new Error(`tensor exceeds stage ${entry.name}`);
    model.tensors.set(entry.name, {
      name: entry.name,
      block: entry.block,
      layer: entry.layer,
      stage: entry.stage,
      stageOffset: entry.stageOffset,
      byteLength: entry.byteLength,
      bytes: stage.subarray(entry.stageOffset, entry.stageOffset + entry.byteLength),
    });
  }
  for (const stage of manifest.stages) {
    const bytes = stages.get(stage.id)!;
    const padded = new Uint8Array(Math.ceil((bytes.byteLength + 4) / 4) * 4);
    padded.set(bytes);
    const buffer = device.createBuffer({
      label: `stage ${stage.id}`,
      size: padded.byteLength,
      usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, padded);
    model.bytesUploaded += bytes.byteLength;
    model.stages.set(stage.id, buffer);
  }
  return model;
}

/**
 * Load the weights for the reference backend once, on `renderer`'s device. Pass the result as `model` to every
 * `referenceWgslBackend.create` (resizes rebuild the graph, not the weights), and dispose it when done.
 */
export async function loadReferenceModel(
  renderer: any,
  source: NRBackendModelSource,
  onProgress?: (message: string) => void,
): Promise<ReferenceModel> {
  const device = rendererDevice(renderer);
  let model: any;
  if (typeof source === 'string' || source instanceof URL) {
    // The upstream loader, as is: it fetches `${directory}/manifest.json` and `${directory}/model/<file>`.
    const directory = String(source).replace(/\/+$/, '');
    model = await new Model(device).load(directory, (loaded: number, total: number) =>
      onProgress?.(`loading weights ${(loaded / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MiB`),
    );
  } else if ('stages' in source) {
    model = modelFromStages(device, source.manifest, source.stages);
  } else {
    const manifestBytes = source.files.get('manifest.json');
    const manifest: NRManifestLike =
      source.manifest ??
      (manifestBytes ? JSON.parse(new TextDecoder().decode(manifestBytes)) : undefined) ??
      (() => {
        throw new Error('the in-memory model has no manifest.json');
      })();
    const stages = new Map<string, Uint8Array>();
    for (const stage of manifest.stages) {
      const bytes = source.files.get(`model/${stage.file}`);
      if (bytes) stages.set(stage.id, bytes);
    }
    model = modelFromStages(device, manifest, stages);
  }
  return {
    isReferenceModel: true,
    device,
    model,
    get bytesUploaded() {
      return model.bytesUploaded as number;
    },
    dispose: () => model.destroy(),
  };
}

// -------------------------------------------------------------------------------------------------------------------
// GPU-resident I/O shared with three.js
// -------------------------------------------------------------------------------------------------------------------

/**
 * Make three.js create `tensor`'s GPU buffer now and return it. three's attribute code creates a buffer only when the
 * attribute has none yet (`WebGPUAttributeUtils.createAttribute`), so a later TSL binding of the same attribute reuses
 * this buffer: frame kernels in TSL and the upstream WGSL graph then share one GPUBuffer, no copies.
 */
export function sharedTensorBuffer(renderer: any, tensor: NRTensor): GPUBuffer {
  renderer.backend.createStorageAttribute(tensor.attribute);
  const buffer: GPUBuffer | undefined = renderer.backend.get(tensor.attribute).buffer;
  if (!buffer) throw new Error(`three did not create a GPU buffer for ${tensor.label}`);
  return buffer;
}

/** The reference's tensor record (`passes.js` `Tensors.allocate`) over a buffer three owns. */
const referenceTensor = (tensor: NRTensor, buffer: GPUBuffer) => ({
  label: tensor.label,
  rows: tensor.rows,
  channels: tensor.channels,
  format: tensor.format,
  allocRows: tensor.allocRows,
  buffer,
  byteLength: tensor.byteLength,
  validBytes: tensor.validBytes,
});

const tensorKey = (tensor: NRTensor) => `${tensor.label}/${tensor.rows}x${tensor.channels}/${tensor.format}`;

// -------------------------------------------------------------------------------------------------------------------
// The network
// -------------------------------------------------------------------------------------------------------------------

/** Extra options of the reference backend (on top of `NRBackendCreateOptions`). */
export interface ReferenceWgslCreateOptions extends Omit<NRBackendCreateOptions, 'model'> {
  /** The weights, or a model from `loadReferenceModel` (shared, not disposed with the backend). */
  model: NRBackendModelSource | ReferenceModel;
  /** Extra WGSL to compile against the shared layout, as the upstream `extraShaders` (text, not paths). */
  extraShaders?: readonly { name: string; code: string; entryPoints: readonly string[] }[];
  /** Called after the input features exist and before the graph is recorded (upstream `before`). */
  before?: (network: any) => void;
  /** Called after the graph is recorded and before pipelines are finished (upstream `after`). */
  after?: (network: any) => void;
}

interface ReferencePrograms {
  kernels: any;
  matmul: any;
  window: any;
}

// The upstream compiler keys pipelines by module and layout identity plus specialization constants.
// Recreating those objects on resize made every old pipeline a cache miss. Keep the immutable programs,
// not graph buffers or temporal history. ViT token padding and extra shader source select a program set;
// the upstream cache still distinguishes each GEMM/window shape and all its weight-layout constants.
const devicePrograms = new WeakMap<GPUDevice, Map<string, Promise<ReferencePrograms>>>();

function referencePrograms(
  device: GPUDevice,
  paddedVitTokens: number,
  extraShaders: NonNullable<ReferenceWgslCreateOptions['extraShaders']>,
): Promise<ReferencePrograms> {
  let cache = devicePrograms.get(device);
  if (!cache) {
    cache = new Map();
    devicePrograms.set(device, cache);
    void device.lost.then(() => devicePrograms.delete(device));
  }
  const key = JSON.stringify([
    paddedVitTokens,
    extraShaders.map(({ name, code, entryPoints }) => [name, code, entryPoints]),
  ]);
  const existing = cache.get(key);
  if (existing) return existing;
  const programs = (async () => {
    const numerics = SHADERS['numerics.wgsl'];
    const kernels = await Kernels.create(device);
    await kernels.add(numerics, SHADERS['gemm_f16.wgsl'], 'gemm_f16.wgsl', ['gemm_f16']);
    await kernels.add(numerics, SHADERS['vit.wgsl'], 'vit.wgsl', ['vit_normalize', 'vit_attend'], {
      PADDED_TOKENS: paddedVitTokens,
    });
    await kernels.add(numerics, SHADERS['ops.wgsl'], 'ops.wgsl', [
      'convert_f32_to_f16',
      'downsample',
      'upsample_residual',
      'post_blend',
    ]);
    await kernels.add(numerics, SHADERS['preprocess.wgsl'], 'preprocess.wgsl', ['preprocess']);
    for (const { name, code, entryPoints } of extraShaders) {
      await kernels.add(numerics, code, name, entryPoints);
    }
    return { kernels, matmul: await Matmul.create(device), window: WindowAttention.create(device, numerics) };
  })();
  cache.set(key, programs);
  const currentCache = cache;
  void programs.catch(() => {
    if (currentCache.get(key) === programs) currentCache.delete(key);
  });
  return programs;
}

/**
 * The reference network on a three.js renderer. `network` is the upstream `Network` instance (its `run`,
 * `readHead`, `readBoundary`, `readTensorByLabel` and `destroy` are the upstream methods).
 */
export class ReferenceWgslBackend implements NRBackend {
  readonly id = 'reference-wgsl' as const;
  readonly label = 'OpenDLSS-NR reference WebGPU (WGSL, upstream code)';
  readonly requirements = REFERENCE_REQUIREMENTS;
  readonly renderer: any;
  readonly device: GPUDevice;
  /** The upstream `Network` (vendored src/network.js). */
  readonly network: any;
  readonly geometry: NRBackendGeometry;
  readonly features: NRTensor;
  readonly head: NRTensor;
  /** Milliseconds `create` took (pipeline compilation dominates). */
  readonly createMilliseconds: number;
  private readonly referenceModel: ReferenceModel;
  private readonly ownsModel: boolean;
  private readonly timer: NRFrameTimer;
  private disposed = false;

  private constructor(fields: {
    renderer: any;
    network: any;
    features: NRTensor;
    head: NRTensor;
    referenceModel: ReferenceModel;
    ownsModel: boolean;
    createMilliseconds: number;
  }) {
    this.renderer = fields.renderer;
    this.device = fields.network.device;
    this.network = fields.network;
    const g = fields.network.geometry;
    this.geometry = {
      validWidth: g.validWidth,
      validHeight: g.validHeight,
      fullWidth: g.fullWidth,
      fullHeight: g.fullHeight,
      fullRows: g.fullRows,
    };
    this.features = fields.features;
    this.head = fields.head;
    this.referenceModel = fields.referenceModel;
    this.ownsModel = fields.ownsModel;
    this.createMilliseconds = fields.createMilliseconds;
    this.timer = new NRFrameTimer(this.device);
  }

  /** `Network.create` of the upstream (src/network.js), on three's device, with the WGSL from the vendored strings. */
  static async create(options: ReferenceWgslCreateOptions): Promise<ReferenceWgslBackend> {
    const started = performance.now();
    const { renderer, width, height, captureBoundaries = false, onProgress } = options;
    const device = rendererDevice(renderer);
    const reason = referenceUnavailableReason(device);
    if (reason) throw new Error(reason);
    requireWorkgroupStorage(device);

    const network = new Network();
    network.device = device;
    network.geometry = geometryFromValid(width, height);
    const geometry = network.geometry;

    onProgress?.('compiling kernels');
    const { kernels, matmul, window } = await referencePrograms(
      device,
      geometry.paddedVitTokens,
      options.extraShaders ?? [],
    );
    network.kernels = kernels;
    network.matmul = matmul;
    network.window = window;

    let referenceModel: ReferenceModel;
    let ownsModel = false;
    if (isReferenceModel(options.model)) {
      if (options.model.device !== device) throw new Error('the reference model was loaded on another device');
      referenceModel = options.model;
    } else {
      onProgress?.('loading weights');
      referenceModel = await loadReferenceModel(renderer, options.model, onProgress);
      ownsModel = true;
    }
    network.model = referenceModel.model;
    // Upstream: a borrowed model outlives the graph (`destroy` leaves it and the device alone). Always true here:
    // the device is three's, and an owned model is disposed by this class.
    network.borrowedModel = true;

    onProgress?.('recording the graph');
    network.tensors = new Tensors(device);
    // The two tensors frame kernels touch are three storage attributes; the upstream graph gets their buffers under
    // the keys it allocates them with, so `Tensors.allocate` returns them instead of making its own.
    const features = createTensor('input features', geometry.fullRows, 16, 'f32');
    const head = createTensor('head', geometry.fullRows, 4, 'f32');
    for (const tensor of [features, head]) {
      network.tensors.byKey.set(tensorKey(tensor), referenceTensor(tensor, sharedTensorBuffer(renderer, tensor)));
    }
    network.features = network.tensors.byKey.get(tensorKey(features));
    network.graph = new Graph(
      {
        device,
        kernels,
        matmul: network.matmul,
        window: network.window,
        tensors: network.tensors,
        model: network.model,
        geometry,
      },
      { captureBoundaries },
    );
    network.recorder = new Recorder(device, kernels, network.tensors);
    options.before?.(network);
    network.graph.record(network.recorder, network.features);
    options.after?.(network);
    if (network.graph.head !== network.tensors.byKey.get(tensorKey(head))) {
      throw new Error('the reference graph did not adopt the shared head tensor');
    }
    // Counts recorded dispatches, including repeated/cached pipelines; it is not a cold-compile count.
    await network.recorder.finish((done: number, count: number) => onProgress?.(`preparing kernels ${done}/${count}`));
    onProgress?.(
      `ready: ${network.recorder.dispatchCount} dispatches, ` +
        `${(network.tensors.total / 1048576).toFixed(0)} MiB of activations, ` +
        `${(network.model.bytesUploaded / 1048576).toFixed(0)} MiB of weights`,
    );
    return new ReferenceWgslBackend({
      renderer,
      network,
      features,
      head,
      referenceModel,
      ownsModel,
      createMilliseconds: performance.now() - started,
    });
  }

  get dispatchCount(): number {
    return this.network.recorder.dispatchCount;
  }

  get boundaryNames(): readonly string[] {
    return this.network.boundaryNames;
  }

  get memory(): NRBackendMemory {
    // `Tensors.total` counts what it allocated; the two shared tensors were put in by hand.
    return {
      activationBytes: this.network.tensors.total + this.features.byteLength + this.head.byteLength,
      weightBytes: this.referenceModel.bytesUploaded,
    };
  }

  /** The upstream blend scale (`Model.blendScale`, block 70) the compose kernel needs. */
  get blendScale(): number {
    return this.network.model.blendScale();
  }

  writeFeatures(data: Float32Array): void {
    this.checkAlive();
    if (data.length !== this.geometry.fullRows * 16) {
      throw new RangeError(`writeFeatures: expected ${this.geometry.fullRows * 16} floats, got ${data.length}`);
    }
    this.network.writeFeatures(data);
  }

  async run(options: NRRunOptions = {}): Promise<NRFrameTiming> {
    this.checkAlive();
    const { until } = options;
    // The first frame goes through the upstream `Network.run`, which validates it inside an error scope and waits;
    // its timing is wall time only. Later frames are encoded and submitted exactly as `run` does, without the wait,
    // so the timer's closing timestamp lands right behind the frame on the queue.
    if (!this.network.validated) {
      return this.timer.measure(() => this.withCut(until, () => this.network.run()), { gpu: false });
    }
    return this.timer.measure(
      () =>
        this.withCut(until, () => {
          const encoder = this.device.createCommandEncoder({ label: 'nr frame' });
          this.network.recorder.encode(encoder);
          this.device.queue.submit([encoder.finish()]);
        }),
      { gpu: options.timing ?? false },
    );
  }

  /**
   * Bisection: run `body` with the recorded items cut after the `until`-th dispatch (copies in between stay), as the
   * design's `after: (net) => { net.recorder.passes.length = cut; }`, but reversible.
   */
  private async withCut(until: number | undefined, body: () => unknown): Promise<void> {
    if (until === undefined) {
      await body();
      return;
    }
    const recorder = this.network.recorder;
    const passes = recorder.passes;
    let dispatches = 0;
    let cut = 0;
    while (cut < passes.length && (passes[cut].kind !== 'dispatch' || dispatches < until)) {
      if (passes[cut].kind === 'dispatch') dispatches += 1;
      cut += 1;
    }
    recorder.passes = passes.slice(0, cut);
    try {
      await body();
    } finally {
      recorder.passes = passes;
    }
  }

  async readHead(): Promise<Float32Array> {
    this.checkAlive();
    return this.network.readHead();
  }

  async readBoundary(name: string): Promise<Uint8Array> {
    this.checkAlive();
    return this.network.readBoundary(name);
  }

  async readTensor(label: string): Promise<Uint8Array> {
    this.checkAlive();
    return (await this.network.readTensorByLabel(label)).bytes;
  }

  /** Read any buffer of this device (e.g. a frame buffer of `ReferenceFrame`). */
  async readBuffer(buffer: GPUBuffer, byteLength: number): Promise<Uint8Array> {
    return new Uint8Array(await readBack(this.device, buffer, byteLength));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    // The shared tensors belong to three: take them out before the upstream `Tensors.destroy` frees the rest.
    for (const tensor of [this.features, this.head]) {
      this.network.tensors.byKey.delete(tensorKey(tensor));
      this.renderer.backend.destroyAttribute(tensor.attribute);
    }
    this.network.destroy();
    this.timer.dispose();
    if (this.ownsModel) this.referenceModel.dispose();
  }

  private checkAlive(): void {
    if (this.disposed) throw new Error('this reference-wgsl backend was disposed');
  }
}

/** The factory: list it next to the TSL backend's in a backend switch. */
export const referenceWgslBackend: NRBackendFactory<ReferenceWgslBackend> = {
  id: 'reference-wgsl',
  label: 'OpenDLSS-NR reference WebGPU (WGSL, upstream code)',
  requirements: REFERENCE_REQUIREMENTS,
  unavailableReason: (renderer) => {
    const device: GPUDevice | undefined = renderer?.backend?.device;
    return device ? referenceUnavailableReason(device) : 'the renderer has no WebGPU device';
  },
  create: (options) => ReferenceWgslBackend.create(options),
};
