// The network as a three.js object: weights, a recorded graph for one resolution, and one frame per compute pass.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Plays the role of the reference's
// ports/browser-webgpu/src/network.js `Network`: everything expensive happens once (the weights are loaded and re-laid
// out, the graph is recorded for one valid size and its kernels compiled), and a frame is then one
// `renderer.compute([...])` call - one command encoder, one compute pass, 451 dispatches (plus one word copy per
// boundary when capturing them).
//
// Not affiliated with NVIDIA. "DLSS" is an NVIDIA trademark used descriptively; no NVIDIA weights are included.

import { alignUp, geometryFromValid, type NRGeometry } from './geometry.js';
import { checkNRDeviceLimits } from './device.js';
import { NRGraph } from './graph/Graph.js';
import { NRModel, type NRModelFiles, type NRModelLoadOptions } from './model/Model.js';
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
   * e.g. across resizes), or an in-memory model directory (`NRModelFiles`). Give this or `modelUrl`.
   */
  model?: NRModel | NRModelFiles;
  /** URL of a model directory (`manifest.json` and `model/stages/*`, the reference's layout). */
  modelUrl?: string | URL;
  /** Valid image size; the padded field (and every level) follows from it (`geometryFromValid`). */
  width: number;
  height: number;
  /** Keep a copy of every block boundary (79), readable with `readBoundary`, for parity checks. */
  captureBoundaries?: boolean;
  onProgress?: (progress: NRProgress) => void;
  /** Compile every kernel during `create` (default true); otherwise the first `run` compiles synchronously. */
  compile?: boolean;
  /** Options for loading `modelUrl` / in-memory model files (SHA-256 verification, custom fetch). */
  load?: Omit<NRModelLoadOptions, 'onProgress'>;
}

export interface NRRunOptions {
  /**
   * Run only the first `until` dispatches of the frame (0-451), each with its capture: the reference's bisection
   * (truncate its recorder's passes at the same index) for finding the first dispatch where two runs diverge.
   */
  until?: number;
}

const isModel = (value: unknown): value is NRModel => value instanceof NRModel;

/** The network for one resolution. */
export class NRNetwork {
  readonly renderer: any;
  readonly model: NRModel;
  readonly geometry: NRGeometry;
  readonly graph: NRGraph;
  private readonly borrowedModel: boolean;
  private validated = false;
  private disposed = false;

  private constructor(renderer: any, model: NRModel, borrowedModel: boolean, graph: NRGraph) {
    this.renderer = renderer;
    this.model = model;
    this.borrowedModel = borrowedModel;
    this.graph = graph;
    this.geometry = graph.geometry;
  }

  /** Load the weights (unless given), record the graph for `width x height`, and compile its kernels. */
  static async create(options: NRNetworkOptions): Promise<NRNetwork> {
    const { renderer, width, height, onProgress } = options;
    if (!renderer) throw new Error('NRNetwork.create needs a WebGPURenderer');
    if (renderer.hasInitialized && !renderer.hasInitialized()) await renderer.init();
    const device: GPUDevice | undefined = renderer.backend?.device;
    if (!device) throw new Error('NRNetwork needs a WebGPURenderer on the WebGPU backend');
    checkNRDeviceLimits(device);
    const geometry = geometryFromValid(width, height);

    let model: NRModel;
    let borrowed = false;
    if (isModel(options.model)) {
      model = options.model;
      borrowed = true;
    } else {
      const source = options.model ?? options.modelUrl;
      if (!source) throw new Error('NRNetwork.create needs `model` or `modelUrl`');
      onProgress?.({ phase: 'loading', message: 'loading weights' });
      model = await NRModel.load(source, {
        ...options.load,
        onProgress: (loaded, total) =>
          onProgress?.({
            phase: 'loading',
            message: `loading weights ${(loaded / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MiB`,
            loaded,
            total,
          }),
      });
    }

    onProgress?.({ phase: 'recording', message: 'recording the graph' });
    const graph = new NRGraph(model, geometry, { captureBoundaries: options.captureBoundaries ?? false });
    const network = new NRNetwork(renderer, model, borrowed, graph);

    if (options.compile ?? true) {
      // One node at a time (three's own progress callback needs `ProgressEvent`, which Node lacks); programs are
      // shared by WGSL text, so most nodes after the first of a shape only build bindings.
      const nodes = graph.passes.map((pass) => pass.kernel.node);
      for (const [index, node] of nodes.entries()) {
        await renderer.compileComputeAsync(node);
        onProgress?.({
          phase: 'compiling',
          message: `compiling kernels ${index + 1}/${nodes.length}`,
          loaded: index + 1,
          total: nodes.length,
        });
      }
    }
    onProgress?.({
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

  /** Dispatches in a frame (451). */
  get dispatchCount(): number {
    return this.graph.dispatches.length;
  }

  /** Dispatch labels in order (the reference's). */
  get dispatchLabels(): string[] {
    return this.graph.labels;
  }

  /** The captured boundary names in capture order (empty unless `captureBoundaries`). */
  get boundaryNames(): string[] {
    return [...this.graph.boundaries.keys()];
  }

  /** The kernels of one frame (or its first `until` dispatches), in submission order. */
  kernels({ until }: NRRunOptions = {}): NRKernel[] {
    const cut = until ?? this.graph.dispatches.length;
    if (!Number.isInteger(cut) || cut < 0 || cut > this.graph.dispatches.length) {
      throw new RangeError(`until ${until} is not a dispatch count in [0, ${this.graph.dispatches.length}]`);
    }
    return this.graph.passes.filter((pass) => pass.index < cut).map((pass) => pass.kernel);
  }

  /** Replace the input features with `data` (f32 `[fullRows][16]`); uploaded before the next frame. */
  writeFeatures(data: Float32Array): void {
    const expected = this.geometry.fullRows * 16;
    if (data.length !== expected) {
      throw new RangeError(`features hold ${data.length} values; the ${this.fieldSize} field needs ${expected}`);
    }
    writeBuffer(this.graph.features, data);
  }

  /**
   * One frame as one compute pass. Resolves once the GPU has finished. The first full frame (and every truncated one)
   * runs inside a validation error scope: a command buffer WebGPU rejects is dropped whole, and the only symptom
   * would be outputs that never change.
   */
  async run(options: NRRunOptions = {}): Promise<void> {
    this.assertLive();
    const kernels = this.kernels(options);
    if (kernels.length === 0) return;
    const device: GPUDevice = this.renderer.backend.device;
    const checking = !this.validated || options.until !== undefined;
    if (checking) device.pushErrorScope('validation');
    this.renderer.compute(kernels.map((k) => k.node));
    if (checking) {
      const error = await device.popErrorScope();
      if (error) throw new Error(`the frame was rejected: ${error.message}`);
      if (options.until === undefined) this.validated = true;
    }
    await device.queue.onSubmittedWorkDone();
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

  /** The tensor allocated under `label` (first of that label). */
  tensor(label: string): NRTensor {
    const found = this.graph.tensors.byLabel(label)[0];
    if (!found) {
      throw new Error(
        `no tensor labelled "${label}"; have ${[...this.graph.tensors.byKey.values()].map((t) => t.label).join(', ')}`,
      );
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
  }

  private get fieldSize(): string {
    return `${this.geometry.fullWidth}x${this.geometry.fullHeight}`;
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
