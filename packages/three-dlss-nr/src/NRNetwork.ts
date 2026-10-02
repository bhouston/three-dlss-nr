// The network as a three.js object: weights, a recorded graph for one resolution, and one frame per compute pass.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Plays the role of the reference's
// ports/browser-webgpu/src/network.js `Network`: everything expensive happens once (the weights are loaded and re-laid
// out, the graph is recorded for one valid size and its kernels compiled), and a frame is then one
// `renderer.compute([...])` call - one command encoder, one compute pass, 451 dispatches (plus one word copy per
// boundary when capturing them). It implements the backend-neutral `NRBackend` interface (`backend/NRBackend.ts`), so
// it can be swapped with the reference shim (`three-dlss-nr/reference-backend`).
//
// Not affiliated with NVIDIA. "DLSS" is an NVIDIA trademark used descriptively; no NVIDIA weights are included.

import type {
  NRBackend,
  NRBackendFactory,
  NRBackendMemory,
  NRBackendModelSource,
  NRBackendRequirements,
  NRFrameTiming,
  NRModelStagesLike,
  NRRunOptions,
} from './backend/NRBackend.js';
import { NRFrameTimer } from './backend/timing.js';
import { checkNRDeviceLimits, NR_MIN_WORKGROUP_STORAGE, nrDeviceProblems } from './device.js';
import { alignUp, geometryFromValid, type NRGeometry } from './geometry.js';
import { windowQueriesFor } from './graph/attention.js';
import { NRGraph } from './graph/Graph.js';
import type { NRManifest } from './model/manifest.js';
import { NRModel, type NRModelLoadOptions, type NRModelSource } from './model/Model.js';
import { attributeBytes, writeBuffer } from './tensors.js';
import type { NRKernel, NRTensor, StorageBufferAttribute } from './types.js';

/** Progress of `NRNetwork.create`. */
export interface NRProgress {
  phase: 'loading' | 'recording' | 'compiling' | 'ready';
  message: string;
  /** Bytes (loading) or kernels (compiling) done so far. */
  loaded?: number;
  /** Bytes (loading) or kernels (compiling) in all. */
  total?: number;
}

export interface NRNetworkOptions {
  /**
   * An initialized three.js `WebGPURenderer` whose device meets the network's limits (`createNRRenderer` makes one;
   * `checkNRDeviceLimits` checks an existing one).
   */
  renderer: any;
  /**
   * The weights: a loaded `NRModel` (borrowed: the network never disposes it, so one model can serve several networks,
   * e.g. across resizes), a parsed `{ manifest, stages }`, an in-memory model directory (`generateSyntheticModel()`
   * from `three-dlss-nr/synthetic`), or a model directory URL. Give this or `modelUrl`.
   */
  model?: NRModel | NRBackendModelSource;
  /** URL of a model directory (`manifest.json` and `model/stages/*`, the reference's layout). */
  modelUrl?: string | URL;
  /** Valid image size; the padded field (and every level) follows from it (`geometryFromValid`). */
  width: number;
  height: number;
  /** Keep a copy of every block boundary (79), readable with `readBoundary`, for parity checks. */
  captureBoundaries?: boolean;
  /** Progress while loading and compiling: a message, and the same as structured data. */
  onProgress?: (message: string, progress: NRProgress) => void;
  /** Compile every kernel during `create` (default true); otherwise the first `run` compiles synchronously. */
  compile?: boolean;
  /** Options for loading a URL / in-memory model directory (SHA-256 verification, custom fetch). */
  load?: Omit<NRModelLoadOptions, 'onProgress'>;
}

/** What the TSL port needs from a device: no features (it never uses f16), the workgroup storage of its kernels. */
export const TSL_REQUIREMENTS: NRBackendRequirements = {
  features: [],
  limits: {
    maxComputeWorkgroupStorageSize: NR_MIN_WORKGROUP_STORAGE,
    maxComputeInvocationsPerWorkgroup: 256,
    maxStorageBuffersPerShaderStage: 8,
  },
};

const isStages = (value: unknown): value is NRModelStagesLike =>
  typeof value === 'object' && value !== null && 'stages' in value && 'manifest' in value;

/** The network for one resolution: the TSL implementation of `NRBackend`. */
export class NRNetwork implements NRBackend {
  readonly id = 'tsl' as const;
  readonly label = 'three-dlss-nr (TSL)';
  readonly requirements = TSL_REQUIREMENTS;
  readonly renderer: any;
  readonly model: NRModel;
  readonly geometry: NRGeometry;
  readonly graph: NRGraph;
  private readonly borrowedModel: boolean;
  private readonly timer: NRFrameTimer;
  private validated = false;
  private disposed = false;

  private constructor(renderer: any, model: NRModel, borrowedModel: boolean, graph: NRGraph) {
    this.renderer = renderer;
    this.model = model;
    this.borrowedModel = borrowedModel;
    this.graph = graph;
    this.geometry = graph.geometry;
    this.timer = new NRFrameTimer(renderer.backend.device);
  }

  /** Load the weights (unless given), record the graph for `width x height`, and compile its kernels. */
  static async create(options: NRNetworkOptions): Promise<NRNetwork> {
    const { renderer, width, height } = options;
    const progress = (update: NRProgress) => options.onProgress?.(update.message, update);
    if (!renderer) throw new Error('NRNetwork.create needs a WebGPURenderer');
    if (renderer.hasInitialized && !renderer.hasInitialized()) await renderer.init();
    const device: GPUDevice | undefined = renderer.backend?.device;
    if (!device) throw new Error('NRNetwork needs a WebGPURenderer on the WebGPU backend');
    checkNRDeviceLimits(device);
    const geometry = geometryFromValid(width, height);

    let model: NRModel;
    let borrowed = false;
    const source = options.model ?? options.modelUrl;
    if (source instanceof NRModel) {
      model = source;
      borrowed = true;
    } else if (isStages(source)) {
      model = new NRModel(source.manifest as NRManifest, source.stages);
    } else {
      if (!source) throw new Error('NRNetwork.create needs `model` or `modelUrl`');
      progress({ phase: 'loading', message: 'loading weights' });
      model = await NRModel.load(source as NRModelSource, {
        ...options.load,
        onProgress: (loaded, total) =>
          progress({
            phase: 'loading',
            message: `loading weights ${(loaded / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MiB`,
            loaded,
            total,
          }),
      });
    }

    progress({ phase: 'recording', message: 'recording the graph' });
    const graph = new NRGraph(model, geometry, {
      captureBoundaries: options.captureBoundaries ?? false,
      windowQueries: windowQueriesFor(device),
    });
    const network = new NRNetwork(renderer, model, borrowed, graph);

    if (options.compile ?? true) {
      // One node at a time (three's own progress callback needs `ProgressEvent`, which Node lacks). Programs are
      // shared by WGSL text, so most nodes after the first of a shape only build their bindings.
      const nodes = graph.passes.map((pass) => pass.kernel.node);
      for (const [index, node] of nodes.entries()) {
        await renderer.compileComputeAsync(node);
        progress({
          phase: 'compiling',
          message: `compiling kernels ${index + 1}/${nodes.length}`,
          loaded: index + 1,
          total: nodes.length,
        });
      }
    }
    progress({
      phase: 'ready',
      message:
        `ready: ${graph.dispatches.length} dispatches, ` +
        `${(graph.tensors.total / 1048576).toFixed(0)} MiB of activations`,
    });
    return network;
  }

  /** The input features tensor, f32 `[fullRows][16]` (a frame kernel such as `createInputFeatures` writes it). */
  get features(): NRTensor {
    return this.graph.features;
  }

  /** The f32 RGBA head `[fullRows][4]` (rgb residual and blend logit; `createCompose` reads it). */
  get head(): NRTensor {
    return this.graph.head;
  }

  /** Compute dispatches in a frame: 451, plus one word copy per boundary when capturing (`NRBackend`). */
  get dispatchCount(): number {
    return this.graph.passes.length;
  }

  /** Dispatch labels in order (the reference's; captures excluded). */
  get dispatchLabels(): string[] {
    return this.graph.labels;
  }

  /** The captured boundary names in capture order (empty unless `captureBoundaries`). */
  get boundaryNames(): string[] {
    return [...this.graph.boundaries.keys()];
  }

  /** GPU bytes of the activations (boundary copies included) and of the weights this network binds. */
  get memory(): NRBackendMemory {
    let weightBytes = 0;
    for (const attribute of this.graph.weightAttributes) weightBytes += attribute.array.byteLength;
    return { activationBytes: this.graph.tensors.total, weightBytes };
  }

  /** The kernels of one frame (or of its first `until` dispatches, each with its capture), in submission order. */
  kernels({ until }: Pick<NRRunOptions, 'until'> = {}): NRKernel[] {
    const count = this.graph.dispatches.length;
    const cut = until ?? count;
    if (!Number.isInteger(cut) || cut < 0 || cut > count) {
      throw new RangeError(`until ${until} is not a dispatch count in [0, ${count}]`);
    }
    return this.graph.passes.filter((pass) => pass.index < cut).map((pass) => pass.kernel);
  }

  /** Replace the input features with `data` (f32 `[fullRows][16]`); uploaded before the next frame. */
  writeFeatures(data: Float32Array): void {
    this.assertLive();
    const expected = this.geometry.fullRows * 16;
    if (data.length !== expected) {
      const field = `${this.geometry.fullWidth}x${this.geometry.fullHeight}`;
      throw new RangeError(`features hold ${data.length} values; the ${field} field needs ${expected}`);
    }
    // Once three has created the GPU buffer, write it directly: a frame kernel may have written the features on the
    // GPU since, and `needsUpdate` would re-upload the whole CPU copy at an unrelated later point. Before that, the CPU
    // copy is what three uploads when it creates the buffer.
    const gpuBuffer: GPUBuffer | undefined = this.renderer.backend.get(this.graph.features.attribute)?.buffer;
    if (gpuBuffer) this.renderer.backend.device.queue.writeBuffer(gpuBuffer, 0, data);
    else writeBuffer(this.graph.features, data);
  }

  /**
   * One frame as one compute pass (or its first `until` dispatches: bisection against the reference, which truncates
   * its recorder at the same index). Resolves once the GPU has finished, with the frame's timing (`timing: true` adds
   * GPU time from timestamp queries when the device has them). The first full frame, and every truncated one, runs
   * inside a validation error scope: a command buffer WebGPU rejects is dropped whole, and the only symptom would be
   * outputs that never change.
   */
  async run(options: NRRunOptions = {}): Promise<NRFrameTiming> {
    this.assertLive();
    const kernels = this.kernels(options);
    const device: GPUDevice = this.renderer.backend.device;
    const checking = !this.validated || options.until !== undefined;
    let rejected: GPUError | null = null;
    const timing = await this.timer.measure(
      async () => {
        if (kernels.length === 0) return;
        if (checking) device.pushErrorScope('validation');
        this.renderer.compute(kernels.map((k) => k.node));
        if (checking) rejected = await device.popErrorScope();
      },
      { gpu: (options.timing ?? false) && !checking },
    );
    if (rejected) throw new Error(`the frame was rejected: ${(rejected as GPUError).message}`);
    if (checking && options.until === undefined) this.validated = true;
    return timing;
  }

  /** The f32 RGBA head, `[fullRows][4]`. */
  async readHead(): Promise<Float32Array> {
    const bytes = await this.readValid(this.graph.head);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  /** A captured boundary by name (`block-N`, `transition-a-b`, `pooled-a-b`): its valid bytes. */
  async readBoundary(name: string): Promise<Uint8Array> {
    const tensor = this.graph.boundaries.get(name);
    if (!tensor) throw new Error(`no captured boundary ${name}`);
    return this.readValid(tensor);
  }

  /** Any tensor by the label the graph allocated it under (the reference's labels): its valid bytes. */
  async readTensor(label: string): Promise<Uint8Array> {
    return this.readValid(this.tensor(label));
  }

  /** The tensor allocated under `label` (the first of that label). */
  tensor(label: string): NRTensor {
    const found = this.graph.tensors.byLabel(label)[0];
    if (!found) {
      const labels = [...this.graph.tensors.byKey.values()].map((t) => t.label).join(', ');
      throw new Error(`no tensor labelled "${label}"; have ${labels}`);
    }
    return found;
  }

  /** Release the GPU buffers and pipelines of this network (and the model's, unless it was borrowed). */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const pass of this.graph.passes) pass.kernel.node.dispose?.();
    const attributes = this.renderer._attributes;
    const release = (attribute: StorageBufferAttribute) => {
      try {
        attributes?.delete(attribute);
      } catch {
        // never created on the GPU
      }
    };
    for (const tensor of this.graph.tensors.byKey.values()) release(tensor.attribute);
    if (!this.borrowedModel) {
      for (const attribute of this.graph.weightAttributes) release(attribute);
      this.model.dispose();
    }
    this.graph.tensors.clear();
    this.timer.dispose();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('the network is disposed');
  }

  private async readValid(tensor: NRTensor): Promise<Uint8Array> {
    this.assertLive();
    // A buffer no kernel has bound yet (no compile, or a truncated run) exists only on the CPU, zero-filled.
    if (!this.renderer.backend.get(tensor.attribute)?.buffer) {
      return attributeBytes(tensor.attribute).slice(0, tensor.validBytes);
    }
    const buffer: ArrayBuffer = await this.renderer.getArrayBufferAsync(
      tensor.attribute,
      null,
      0,
      alignUp(tensor.validBytes, 4),
    );
    return new Uint8Array(buffer, 0, tensor.validBytes);
  }
}

/** The TSL port as an `NRBackend` factory (the benchmark, the fidelity suite and the demo look for this name). */
export const tslBackend: NRBackendFactory<NRNetwork> = {
  id: 'tsl',
  label: 'three-dlss-nr (TSL)',
  requirements: TSL_REQUIREMENTS,
  unavailableReason(renderer: any): string | null {
    const device: GPUDevice | undefined = renderer?.backend?.device;
    if (!device) return 'the renderer has no WebGPU device';
    const problems = nrDeviceProblems(device);
    return problems.length ? `this WebGPU device cannot run the network: ${problems.join('; ')}` : null;
  },
  create: (options) => NRNetwork.create(options),
};
