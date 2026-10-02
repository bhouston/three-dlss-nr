// The per-kernel reference runner: single dispatches of the OpenDLSS-NR WebGPU port, in Node, on our device.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Drives the reference's own modules
// (reference/OpenDLSS-NR/ports/browser-webgpu/src: passes.js `Kernels` / `Tensors` / `Recorder`, matmul/, window/,
// graph.js) exactly as network.js:55-87 wires them, through a `Graph` whose methods `gemm`, `gemmF16`, `op` and
// `windowAttention` (graph.js:59-145) record one dispatch each. The WGSL is read from the submodule on disk.
//
// The reference's FP8 GEMM, window attention and SiLU tables need `shader-f16` (their modules begin
// `enable f16;`); everything else (gemm_f16, ops, vit, numerics) does not. `RefKernels.f16` says which you have.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { Graph } from '@ref/graph.js';
import { bindLayout, compile, readBack, storage, storageFrom } from '@ref/gpu.js';
import { Matmul } from '@ref/matmul/index.js';
import { Kernels, Recorder, Tensors } from '@ref/passes.js';
import { WindowAttention } from '@ref/window/index.js';

import { alignUp } from '../../src/geometry.js';
import type { TensorFormat } from '../../src/types.js';

/** The reference WebGPU port's directory (`ports/browser-webgpu/`), with a trailing separator. */
export const REFERENCE_PORT_DIR = fileURLToPath(
  new URL('../../../../reference/OpenDLSS-NR/ports/browser-webgpu/', import.meta.url),
);

/** `shaderBase` for the reference's `Network.create` (a `file:` URL the fetch shim serves). */
export const referenceShaderBase = (): URL => pathToFileURL(REFERENCE_PORT_DIR);

/** The text of one of the reference's `shaders/*.wgsl`. */
export const referenceShader = (name: string): string =>
  readFileSync(join(REFERENCE_PORT_DIR, 'shaders', name), 'utf8');

/** Raw bytes of a file under the reference port directory (e.g. `web/fixtures/numerics.bin`). */
export const referenceFile = (path: string): Uint8Array => new Uint8Array(readFileSync(join(REFERENCE_PORT_DIR, path)));

/** The reference's FP8 GEMM and window attention compile only on a device with `shader-f16`. */
export const referenceSupportsF16 = (device: GPUDevice): boolean => device.features.has('shader-f16');

/** A reference activation tensor (`passes.js` `Tensors.allocate`). */
export interface RefTensor {
  label: string;
  rows: number;
  channels: number;
  format: TensorFormat;
  allocRows: number;
  buffer: GPUBuffer;
  byteLength: number;
  validBytes: number;
}

/** A reference FP8 weight descriptor (`model.js` `fp8Matrix`) plus the byte offset of skip scales in its buffer. */
export interface RefFp8Matrix {
  buffer: GPUBuffer;
  byteOffset: number;
  k: number;
  matrixChannels: number;
  batchK: number;
  /** Byte offset of the per-column skip scales in `buffer` (the GEMM's `aux`), when given. */
  aux: number;
}

const asBytes = (data: ArrayBufferView): Uint8Array => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

/** Run `numerics.wgsl` + `selftest.wgsl` entry points on a device, as the reference's selftest.js does. */
export class RefSelftest {
  private constructor(
    private readonly device: GPUDevice,
    private readonly module: GPUShaderModule,
    private readonly layout: GPUBindGroupLayout,
    private readonly pipelineLayout: GPUPipelineLayout,
  ) {}

  static async create(device: GPUDevice): Promise<RefSelftest> {
    const code = `${referenceShader('numerics.wgsl')}\n${referenceShader('selftest.wgsl')}`;
    const module = await compile(device, code, 'selftest.wgsl');
    const layout = bindLayout(device, ['read-only-storage', 'storage'], 'selftest');
    return new RefSelftest(device, module, layout, device.createPipelineLayout({ bindGroupLayouts: [layout] }));
  }

  /** One `case_*` entry point over `count` invocations; returns the u32 results. */
  async run(entryPoint: string, count: number, inputs?: Uint32Array): Promise<Uint32Array> {
    const device = this.device;
    const input = storageFrom(device, inputs ?? new Uint32Array(1), `${entryPoint} inputs`);
    const results = storage(device, count * 4, `${entryPoint} results`);
    const pipeline = await device.createComputePipelineAsync({
      layout: this.pipelineLayout,
      compute: { module: this.module, entryPoint },
    });
    const group = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: results } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readBack(device, results, count * 4));
    input.destroy();
    results.destroy();
    return words;
  }
}

export interface RefKernelsOptions {
  /** The ViT's `PADDED_TOKENS` override (vit.wgsl is compiled once per RefKernels). Default 64. */
  paddedVitTokens?: number;
  /** The adapter's info (defaults to `device.adapterInfo`); used to recognize software rasterizers. */
  adapterInfo?: GPUAdapterInfo;
}

/**
 * Mesa llvmpipe (lavapipe, the CI device) exposes `shader-f16` but folds the f32 -> f16 -> f32 round trip
 * `f32(f16(x))` into `x`, so the reference's half roundings (`round_accumulator`, the end of every FDPA group, the
 * SiLU table builder) never happen; a real f16 rounding is left only where a value is stored as f16 or bitcast. The
 * reference FP8 GEMM then differs from real f16 hardware by one half ulp in ~24% of raw half outputs (CI run
 * 37030840217; on an RTX 3060 Ti in Chrome it equals oracleGemmFp8 exactly, test/browser/run-fp8-gemm-chrome.mjs).
 * The reference's f16 kernels are therefore not a trustworthy oracle there. Set DLSS_NR_TRUST_SOFTWARE_F16=1 to run
 * them anyway.
 */
export const SOFTWARE_F16_REASON =
  'llvmpipe folds f32(f16(x)) round trips, so the reference f16 kernels do not round like f16 hardware ' +
  '(set DLSS_NR_TRUST_SOFTWARE_F16=1 to run anyway)';

/** True for adapters whose f16 arithmetic does not round like hardware (Mesa llvmpipe / lavapipe). */
export function isUntrustedSoftwareF16(info: GPUAdapterInfo | undefined): boolean {
  if (!info || process.env.DLSS_NR_TRUST_SOFTWARE_F16 === '1') return false;
  return /llvmpipe|lavapipe/i.test(`${info.vendor} ${info.architecture} ${info.device} ${info.description}`);
}

/**
 * Records reference dispatches and runs them. Typical use:
 *
 *   const ref = await RefKernels.create(device);
 *   const input = ref.tensor('in', 32, 32, 'e4', bytes);
 *   const out = ref.tensor('out', 32, 32, 'e4');
 *   ref.fill(out, SENTINEL_BYTE);
 *   ref.gemm({ input, weights: ref.fp8Matrix({ bytes, k: 32, n: 32 }), output: out, rows: 32, k: 32, n: 32,
 *              label: 'test' });
 *   await ref.run();
 *   const bytes = await ref.read(out);
 */
export class RefKernels {
  readonly device: GPUDevice;
  /** True when the device has `shader-f16`: `gemm` and `windowAttention` are available. */
  readonly f16: boolean;
  // The reference's objects (untyped JS).
  readonly kernels: any;
  readonly matmul: any;
  readonly window: any;
  readonly tensors: any;
  private recorder: any;
  private graph: any;
  private readonly owned: GPUBuffer[] = [];
  private readonly unavailable = new Map<string, string>();
  readonly numerics: string;

  private constructor(
    device: GPUDevice,
    parts: { kernels: any; matmul: any; window: any; tensors: any; numerics: string },
  ) {
    this.device = device;
    this.f16 = referenceSupportsF16(device);
    this.kernels = parts.kernels;
    this.matmul = parts.matmul;
    this.window = parts.window;
    this.tensors = parts.tensors;
    this.numerics = parts.numerics;
    this.reset();
  }

  static async create(device: GPUDevice, options: RefKernelsOptions = {}): Promise<RefKernels> {
    const numerics = referenceShader('numerics.wgsl');
    const kernels = await Kernels.create(device);
    const unavailable = new Map<string, string>();
    // A module the device cannot compile (e.g. FXC's E_FAIL on D3D12 without DXC) leaves its entry points unavailable
    // instead of failing everything; using one throws the recorded reason.
    const add = async (file: string, entryPoints: string[], constants?: Record<string, number>) => {
      try {
        await kernels.add(numerics, referenceShader(file), file, entryPoints, constants);
      } catch (error) {
        for (const entry of entryPoints) unavailable.set(entry, `${file}: ${(error as Error).message.split('\n')[0]}`);
      }
    };
    await add('gemm_f16.wgsl', ['gemm_f16']);
    await add('vit.wgsl', ['vit_normalize', 'vit_attend'], { PADDED_TOKENS: options.paddedVitTokens ?? 64 });
    await add('ops.wgsl', ['convert_f32_to_f16', 'downsample', 'upsample_residual', 'post_blend']);
    const f16 = referenceSupportsF16(device);
    if (!f16) {
      const reason = 'needs shader-f16, which this device lacks';
      unavailable.set('gemm_fp8', reason);
      unavailable.set('window_attend', reason);
    } else if (isUntrustedSoftwareF16(options.adapterInfo ?? (device as any).adapterInfo)) {
      unavailable.set('gemm_fp8', SOFTWARE_F16_REASON);
      unavailable.set('window_attend', SOFTWARE_F16_REASON);
    }
    const matmul = f16 ? await Matmul.create(device) : null;
    const window = f16 ? WindowAttention.create(device, numerics) : null;
    const ref = new RefKernels(device, { kernels, matmul, window, tensors: new Tensors(device), numerics });
    for (const [entry, reason] of unavailable) ref.unavailable.set(entry, reason);
    return ref;
  }

  /**
   * Why a reference kernel cannot run on this device (`'gemm_fp8'`, `'window_attend'`, or an entry point such as
   * `'gemm_f16'`), or `undefined` when it can. Tests skip on it.
   */
  unavailableReason(kernel: string): string | undefined {
    return this.unavailable.get(kernel);
  }

  private require(kernel: string): void {
    const reason = this.unavailable.get(kernel);
    if (reason) throw new Error(`the reference ${kernel} cannot run on this device: ${reason}`);
  }

  /** Start a new recording (dropping anything recorded and not run). */
  reset(): void {
    this.recorder = new Recorder(this.device, this.kernels, this.tensors);
    this.graph = Object.assign(Object.create(Graph.prototype), {
      device: this.device,
      kernels: this.kernels,
      matmul: this.matmul,
      window: this.window,
      tensors: this.tensors,
      recorder: this.recorder,
      model: null,
      options: {},
      boundaries: new Map(),
    });
  }

  /** An activation tensor (zero-filled, rows padded to 64), optionally initialized with `data` from byte 0. */
  tensor(label: string, rows: number, channels: number, format: TensorFormat, data?: ArrayBufferView): RefTensor {
    const tensor: RefTensor = this.tensors.allocate(label, rows, channels, format);
    if (data) this.write(tensor, data);
    return tensor;
  }

  /** Overwrite bytes of a reference buffer. */
  write(target: RefTensor | GPUBuffer, data: ArrayBufferView, byteOffset = 0): void {
    const buffer = 'buffer' in target ? target.buffer : target;
    const bytes = asBytes(data);
    const padded = new Uint8Array(alignUp(bytes.byteLength, 4));
    padded.set(bytes);
    this.device.queue.writeBuffer(buffer, byteOffset, padded);
  }

  /** Fill a whole tensor with one byte (the sentinel). */
  fill(tensor: RefTensor, byte: number): void {
    this.device.queue.writeBuffer(tensor.buffer, 0, new Uint8Array(tensor.byteLength).fill(byte));
  }

  /** A read-only storage buffer holding `data` (+4 bytes of slack, like a stage buffer). */
  buffer(data: ArrayBufferView, label = 'test buffer'): GPUBuffer {
    const bytes = asBytes(data);
    const padded = new Uint8Array(alignUp(bytes.byteLength + 4, 4));
    padded.set(bytes);
    const buffer = storageFrom(this.device, padded, label);
    this.owned.push(buffer);
    return buffer;
  }

  /**
   * An FP8 matrix in its own stage-like buffer: `bytes` (fragment order, `batches * batchK * n` bytes), then the skip
   * scales (halves) at the next 16-byte boundary when given.
   */
  fp8Matrix({
    bytes,
    k,
    n,
    batchK = k,
    scales,
  }: {
    bytes: Uint8Array;
    k: number;
    n: number;
    batchK?: number;
    scales?: Uint16Array;
  }): RefFp8Matrix {
    if (bytes.byteLength !== k * n) throw new Error(`fp8Matrix: ${bytes.byteLength} bytes for ${k}x${n}`);
    const aux = alignUp(bytes.byteLength, 16);
    const stage = new Uint8Array(aux + (scales ? scales.byteLength : 0));
    stage.set(bytes);
    if (scales) stage.set(asBytes(scales), aux);
    return { buffer: this.buffer(stage, 'fp8 matrix'), byteOffset: 0, k, matrixChannels: n, batchK, aux };
  }

  /**
   * Record one FP8 GEMM (`Graph.gemm`). `weights.k` must be `k * batches`. Pass `residual` (a tensor whose `format`
   * is 'e4' or 'f16') to seed from a skip scaled by the matrix's `aux` scales.
   */
  gemm(args: {
    input: RefTensor;
    weights: RefFp8Matrix;
    output?: RefTensor;
    outputF16?: RefTensor;
    rows: number;
    k: number;
    n: number;
    batches?: number;
    broadcast?: boolean;
    partition?: 0 | 256 | 512 | 1024;
    silu?: boolean;
    residual?: RefTensor | null;
    label: string;
  }): void {
    this.require('gemm_fp8');
    this.graph.gemm({ ...args, aux: args.residual ? args.weights.aux : 0 });
  }

  /** Record one f16 GEMM (`Graph.gemmF16`); `weights` holds halves `[k][paddedN]`. */
  gemmF16(args: {
    input: RefTensor;
    weights: GPUBuffer;
    paddedN: number;
    output?: RefTensor;
    outputF16?: RefTensor;
    outputF32?: RefTensor;
    rows: number;
    k: number;
    n: number;
    label: string;
  }): void {
    this.require('gemm_f16');
    this.graph.gemmF16(args);
  }

  /** Record one ops.wgsl dispatch (`Graph.op`): 'convert_f32_to_f16', 'downsample', 'upsample_residual', 'post_blend'. */
  op(entryPoint: string, args: Record<string, unknown>): void {
    this.require(entryPoint);
    this.graph.op(entryPoint, args);
  }

  /** Record one window attention (`Graph.windowAttention`); `prior` / `scales` are buffers. */
  windowAttention(args: {
    qkv: RefTensor;
    attended: RefTensor;
    prior: GPUBuffer;
    scales: GPUBuffer;
    width: number;
    height: number;
    heads: number;
    phase: number;
    label: string;
  }): void {
    this.require('window_attend');
    this.graph.windowAttention(args);
  }

  /** Record one vit.wgsl dispatch, as graph.js `vit()` does; params `[tokens, heads, channels, paddedTokens]`. */
  vit(
    entryPoint: 'vit_normalize' | 'vit_attend',
    buffers: Record<number, GPUBuffer>,
    { tokens, heads, channels, paddedTokens, label }: Record<string, number> & { label?: string },
    workgroups: number[],
  ): void {
    this.require(entryPoint);
    const params = new Uint32Array([tokens, heads, channels, paddedTokens]);
    this.recorder.pass(entryPoint, buffers, params, workgroups, label ?? entryPoint);
  }

  /** The recorder (for direct `pass` / `specialized` use). */
  get currentRecorder(): any {
    return this.recorder;
  }

  /**
   * Compile what was recorded, run it in one compute pass inside a validation error scope, wait for the GPU, and
   * start a new recording.
   */
  async run(): Promise<void> {
    const recorder = this.recorder;
    await recorder.finish();
    this.device.pushErrorScope('validation');
    const encoder = this.device.createCommandEncoder({ label: 'reference kernels' });
    recorder.encode(encoder);
    this.device.queue.submit([encoder.finish()]);
    const error = await this.device.popErrorScope();
    await this.device.queue.onSubmittedWorkDone();
    recorder.paramsBuffer?.destroy();
    this.reset();
    if (error) throw new Error(`reference dispatches rejected: ${error.message}`);
  }

  /** Read a tensor back: its `validBytes` by default, or the whole allocation. */
  async read(tensor: RefTensor, { validOnly = true }: { validOnly?: boolean } = {}): Promise<Uint8Array> {
    const length = validOnly ? tensor.validBytes : tensor.byteLength;
    return new Uint8Array(await readBack(this.device, tensor.buffer, length));
  }

  destroy(): void {
    this.tensors.destroy();
    for (const buffer of this.owned) buffer.destroy();
    this.owned.length = 0;
  }
}
