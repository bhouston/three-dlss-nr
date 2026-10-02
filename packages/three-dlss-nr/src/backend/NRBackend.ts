// The backend-neutral network interface: one NR network on a three.js WebGPURenderer, whichever implementation runs it.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41).
//
// Two implementations exist (or will):
//   * 'tsl'            - the native port: the network as three.js TSL compute nodes (`NRNetwork`, design chunk E);
//   * 'reference-wgsl' - the "shim": OpenDLSS-NR's own WebGPU port (its JavaScript and WGSL, unchanged) driven on
//                        three's GPUDevice (`renderer.backend.device`); entry point `three-dlss-nr/reference-backend`.
// Both take the same model, the same input features and produce the same f32 head and the same block boundaries,
// so a caller (the website demo, the fidelity suite, the benchmark) can swap them at runtime and compare speed and
// output on identical inputs.
//
// The interface follows the `NRNetwork` API of the design (chunk E): `create({ renderer, model, width, height,
// captureBoundaries, onProgress })`, `writeFeatures`, `run({ until })`, `readHead`, `readBoundary`, `readTensor`,
// `dispose`. What it adds: an `id`, the device `requirements`, the input and head as GPU-resident tensors (so frame
// kernels write and read them with no CPU round trip, whichever backend runs), a timing result from `run`, and a
// memory report. `NRNetwork` can implement it as is: `class NRNetwork implements NRBackend` plus a factory object
// `tslBackend: NRBackendFactory<NRNetwork>` exported from the package index (the benchmark and fidelity suite look
// for that name).

import type { NRTensor } from '../types.js';

/** Which implementation runs the network. */
export type NRBackendId = 'tsl' | 'reference-wgsl';

/** What a backend needs from the renderer's GPUDevice. */
export interface NRBackendRequirements {
  /** Device features it cannot run without (the reference needs `shader-f16`; the TSL port needs none). */
  readonly features: readonly GPUFeatureName[];
  /** Minimum device limits. */
  readonly limits: Readonly<Partial<Record<NRBackendLimit, number>>>;
}

/** Device limits a backend may require. */
export type NRBackendLimit =
  | 'maxComputeWorkgroupStorageSize'
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxStorageBuffersPerShaderStage'
  | 'maxStorageBufferBindingSize'
  | 'maxBufferSize';

/**
 * The model manifest fields every backend reads (`docs/weights.md` of OpenDLSS-NR; chunk A's `NRManifest` is a
 * superset).
 */
export interface NRManifestLike {
  readonly totals: { readonly blockCount: number };
  readonly stages: readonly { readonly id: string; readonly file: string; readonly packedByteLength: number }[];
  readonly tensors: readonly {
    readonly name: string;
    readonly block: number;
    readonly layer: number;
    readonly stage: string;
    readonly stageOffset: number;
    readonly byteLength: number;
  }[];
}

/** A model directory held in memory: files keyed by path relative to the directory (`manifest.json`, `model/...`). */
export interface NRModelFilesLike {
  readonly manifest?: NRManifestLike;
  readonly files: ReadonlyMap<string, Uint8Array>;
}

/** A parsed model: its manifest and the stage bytes by stage id (chunk A's loaded `NRModel` has this shape). */
export interface NRModelStagesLike {
  readonly manifest: NRManifestLike;
  readonly stages: ReadonlyMap<string, Uint8Array>;
}

/**
 * Where the weights come from: a model directory URL (`<url>/manifest.json`, `<url>/model/<stage file>`), an
 * in-memory directory (e.g. `generateSyntheticModel()`), or an already parsed model. Load once and pass the parsed
 * model to every backend you create, so switching backends does not re-download 141 MiB.
 */
export type NRBackendModelSource = string | URL | NRModelFilesLike | NRModelStagesLike;

export interface NRBackendCreateOptions {
  /** An initialized three.js `WebGPURenderer` (`await renderer.init()`); the backend runs on its device. */
  renderer: any;
  /** The weights. */
  model: NRBackendModelSource;
  /** The valid (rendered) image size; the padded field follows from it (`geometryFromValid`). */
  width: number;
  height: number;
  /** Keep a copy of every block output for `readBoundary` (parity and fidelity work; costs memory and copies). */
  captureBoundaries?: boolean;
  /** Progress messages while weights load and kernels compile. */
  onProgress?: (message: string) => void;
}

/** The padded field the network runs on (the subset of `NRGeometry` every backend reports). */
export interface NRBackendGeometry {
  readonly validWidth: number;
  readonly validHeight: number;
  readonly fullWidth: number;
  readonly fullHeight: number;
  /** `fullWidth * fullHeight`: rows of the feature and head tensors. */
  readonly fullRows: number;
}

/** How a frame was timed. Both backends use the same method on the same device, so their numbers compare. */
export type NRTimingMethod = 'timestamp-query' | 'submitted-work-done';

/** The cost of one `run`. */
export interface NRFrameTiming {
  readonly method: NRTimingMethod;
  /**
   * GPU time from just before the frame's first command to just after its last (timestamp writes on two empty
   * compute passes submitted around the frame), in ms; `null` when `timestamp-query` is unavailable or not asked for.
   */
  readonly gpuMilliseconds: number | null;
  /** Wall time from the first submit to `queue.onSubmittedWorkDone()` resolving, in ms (includes CPU encode). */
  readonly wallMilliseconds: number;
}

/** Bytes a backend holds on the GPU. */
export interface NRBackendMemory {
  /** Activation tensors (including captured boundaries). */
  readonly activationBytes: number;
  /** Weights and tables as uploaded / re-laid out by this backend. */
  readonly weightBytes: number;
}

export interface NRRunOptions {
  /** Run only the first `until` dispatches of the frame (bisection against the other backend). */
  until?: number;
  /** Measure GPU time with timestamp queries when the device has them (default false: wall time only). */
  timing?: boolean;
}

/** One network at one resolution on one renderer. Rebuild (dispose + create) on resize. */
export interface NRBackend {
  readonly id: NRBackendId;
  /** Human-readable name for UIs and reports. */
  readonly label: string;
  readonly requirements: NRBackendRequirements;
  readonly geometry: NRBackendGeometry;
  /** Compute dispatches per frame (451 for the 71-block network, plus boundary copies when capturing). */
  readonly dispatchCount: number;
  /**
   * The input: f32 `[fullRows][16]` features, GPU-resident. Frame kernels (TSL nodes or raw WGSL) write it in place
   * before `run`; its attribute is bound to the backend's own GPU buffer, so no copy happens.
   */
  readonly features: NRTensor;
  /** The output: f32 `[fullRows][4]` head (rgb residual, blend logit), GPU-resident, valid after `run`. */
  readonly head: NRTensor;
  /** Names of the captured boundaries (empty unless `captureBoundaries`); `block-N`, `transition-a-b`, ... */
  readonly boundaryNames: readonly string[];
  readonly memory: NRBackendMemory;
  /** Replace the input features from the CPU (`Float32Array` of `fullRows * 16`); takes effect on the next `run`. */
  writeFeatures(data: Float32Array): void;
  /** Run one frame; resolves when the GPU has finished it. */
  run(options?: NRRunOptions): Promise<NRFrameTiming>;
  /** The f32 head, `[fullRows][4]`. */
  readHead(): Promise<Float32Array>;
  /** A captured boundary's valid bytes (E4M3 codes `[rows][channels]`). */
  readBoundary(name: string): Promise<Uint8Array>;
  /** Any intermediate tensor by the reference's label (e.g. `'post merge'`), valid bytes. */
  readTensor(label: string): Promise<Uint8Array>;
  /** Release the backend's GPU resources (not the renderer, its device, or a model passed in parsed). */
  dispose(): void;
}

/** Creates backends of one kind; lets a UI list, check and switch implementations. */
export interface NRBackendFactory<B extends NRBackend = NRBackend> {
  readonly id: NRBackendId;
  readonly label: string;
  readonly requirements: NRBackendRequirements;
  /** Why this backend cannot run on `renderer`'s device (a sentence naming the fix), or `null` when it can. */
  unavailableReason(renderer: any): string | null;
  /** Load (or adopt) the weights, compile the kernels, record the graph. Throws `unavailableReason` if set. */
  create(options: NRBackendCreateOptions): Promise<B>;
}

/** The problems that keep a device from meeting `requirements` (empty when it can run). */
export function unmetRequirements(device: GPUDevice, requirements: NRBackendRequirements): string[] {
  const problems: string[] = [];
  for (const feature of requirements.features) {
    if (!device.features.has(feature)) problems.push(`the device lacks the '${feature}' feature`);
  }
  const limits = device.limits as unknown as Record<string, number>;
  for (const [name, minimum] of Object.entries(requirements.limits)) {
    if (minimum !== undefined && !(limits[name] >= minimum)) {
      problems.push(`${name} is ${limits[name]}; needs at least ${minimum}`);
    }
  }
  return problems;
}

/** The GPUDevice of a three.js WebGPURenderer, or a clear error. */
export function rendererDevice(renderer: any): GPUDevice {
  const device: GPUDevice | undefined = renderer?.backend?.device;
  if (!device) {
    throw new Error(
      'the renderer has no WebGPU device: pass an initialized WebGPURenderer (`await renderer.init()`) that did not ' +
        'fall back to WebGL',
    );
  }
  return device;
}
