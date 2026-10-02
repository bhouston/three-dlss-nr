// Building one network dispatch as a three.js TSL compute node.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Plays the role of the reference's
// ports/browser-webgpu/src/passes.js `Recorder.pass` / `specialized`: one dispatch with explicit workgroup counts and
// explicitly typed buffers. See src/README-internals.md for how to write a kernel with it.
//
// What it enforces (design section 3):
//   R6  Only nodes reachable from the kernel body are emitted; a value never assigned silently disappears. A kernel
//       declares the tensors it writes (`writes`) so tests can prefill them with a sentinel and compare them whole.
//   R7  Two `storage()` nodes on one attribute become two bindings, and two writable bindings of one buffer make
//       WebGPU reject the whole command buffer. `kernel()` creates exactly one storage node per attribute, binds
//       inputs read-only, and refuses one attribute declared twice or as both input and output.
//   R12 Dispatches are always explicit `[x, y, z]` workgroup counts (never a count, which would add a uniform and an
//       `instanceIndex >= count` early return); index from `workgroupId` / `localId` (`foldedGroupY`).

import { Fn, storage, workgroupId } from 'three/tsl';

import type { Dim3, NRKernel, NRTensor, StorageBufferAttribute } from '../types.js';
import { u, type TSLNode } from './packed.js';

/** Anything backed by one storage attribute: a tensor, a weight matrix, a vector. */
export interface BufferSource {
  readonly attribute: StorageBufferAttribute;
}

/** Storage nodes handed to the body, by the names the kernel declared them under. */
export type BufferViews<I, O> = { readonly [K in keyof I | keyof O]: TSLNode };

export interface KernelOptions<I extends Record<string, BufferSource>, O extends Record<string, BufferSource>> {
  /** The reference's dispatch label. */
  label: string;
  /** Kernel family (the reference's entry point name: `'gemm_fp8'`, `'window_attend'`, `'downsample'`, ...). */
  kind: string;
  /** `@workgroup_size`; 1-3 components, padded with 1. */
  workgroupSize: readonly number[];
  /** Workgroup counts; 1-3 components, padded with 1. Each must be in [1, 65535]. */
  dispatch: readonly number[];
  /**
   * Buffers bound read-only (`var<storage, read> array<u32>`), by name (alphanumeric; the WGSL binding is `nr_<name>`).
   * Tensors among them become `reads`.
   */
  inputs: I;
  /** Buffers bound read-write (`var<storage, read_write> array<u32>`). Tensors among them become `writes`. */
  outputs: O;
  /**
   * Builds the kernel body; runs inside a TSL `Fn` when three first builds the node (lazily). Receives one
   * `storage(attribute, 'uint')` node per declared buffer. Do not `Return()` early when the kernel uses barriers;
   * bounds-test with `If` / `select` instead.
   */
  body: (buffers: BufferViews<I, O>) => void;
}

const MAX_GROUPS = 65535;

const toDim3 = (values: readonly number[], what: string): Dim3 => {
  if (values.length < 1 || values.length > 3) throw new RangeError(`${what} must have 1 to 3 components`);
  const [x, y = 1, z = 1] = values;
  for (const value of [x, y, z]) {
    if (!Number.isInteger(value) || value < 1) throw new RangeError(`${what}: ${value} is not a positive integer`);
  }
  return [x, y, z];
};

const isTensor = (source: BufferSource): source is NRTensor =>
  'format' in source && 'rows' in source && 'channels' in source;

/**
 * Build one dispatch. Returns an `NRKernel` whose `node` carries its own dispatch size; run a list of them in one
 * compute pass with `renderer.compute(kernels.map((k) => k.node))` (or `runKernels`).
 */
export function kernel<I extends Record<string, BufferSource>, O extends Record<string, BufferSource>>(
  options: KernelOptions<I, O>,
): NRKernel {
  const { label } = options;
  const workgroupSize = toDim3(options.workgroupSize, `${label}: workgroupSize`);
  const dispatch = toDim3(options.dispatch, `${label}: dispatch`);
  for (const count of dispatch) {
    if (count > MAX_GROUPS) throw new RangeError(`${label}: dispatch ${dispatch.join('x')} exceeds 65535 groups`);
  }

  const seen = new Map<StorageBufferAttribute, string>();
  const views: Record<string, TSLNode> = {};
  const reads: NRTensor[] = [];
  const writes: NRTensor[] = [];
  const bind = (name: string, source: BufferSource, writable: boolean): void => {
    if (name in views) throw new Error(`${label}: buffer name "${name}" declared twice`);
    if (!/^[A-Za-z][A-Za-z0-9]*$/.test(name)) throw new Error(`${label}: buffer name "${name}" must be alphanumeric`);
    const attribute = source?.attribute;
    if (!attribute?.isStorageBufferAttribute) {
      throw new TypeError(`${label}: "${name}" is not backed by a StorageBufferAttribute`);
    }
    if (!(attribute.array instanceof Uint32Array)) {
      throw new TypeError(`${label}: "${name}" must hold a Uint32Array (kernels view every buffer as u32 words)`);
    }
    const other = seen.get(attribute);
    if (other !== undefined) {
      throw new Error(`${label}: "${name}" and "${other}" are the same buffer; a kernel binds a buffer once (R7)`);
    }
    seen.set(attribute, name);
    // A fixed binding name (instead of three's `NodeBuffer_<node id>`) keeps the WGSL of two kernels that differ only
    // in their buffers identical, so three compiles one shader module for both (programs are keyed by WGSL text).
    const node = storage(attribute, 'uint', attribute.count).setName(`nr_${name}`);
    views[name] = writable ? node : node.toReadOnly();
    if (isTensor(source)) (writable ? writes : reads).push(source);
  };
  for (const [name, source] of Object.entries(options.inputs)) bind(name, source, false);
  for (const [name, source] of Object.entries(options.outputs)) bind(name, source, true);
  if (Object.keys(options.outputs).length === 0) throw new Error(`${label}: a kernel without outputs does nothing`);

  const frozen = Object.freeze(views) as BufferViews<I, O>;
  const node = Fn(() => {
    options.body(frozen);
  })()
    .compute([...dispatch], [...workgroupSize])
    // three writes the compute node's name into the WGSL (`// flow -> <name>`), so naming it by kind rather than by
    // the per-dispatch label keeps the text - and the compiled program - shared between same-shape dispatches.
    .setName(options.kind);

  return { label, kind: options.kind, node, reads, writes, dispatch, workgroupSize };
}

/**
 * The folded workgroup row index of a dispatch that splits more than 65535 row groups over y and z, as the reference
 * does (`group.y + group.z * 65535`; R12).
 */
export const foldedGroupY = (): TSLNode => workgroupId.y.add(workgroupId.z.mul(u(MAX_GROUPS)));

/** The three.js `WebGPURenderer` (typed `any`: three ships no declarations). */
export type NRRendererLike = any;

/**
 * Run kernels in order in one compute pass. With `validate` (default), the submission is wrapped in a WebGPU
 * validation error scope and a rejected command buffer throws instead of silently leaving every output unchanged (the
 * reference's `Network.run` does the same on its first frame).
 */
export async function runKernels(
  renderer: NRRendererLike,
  kernels: readonly NRKernel[],
  { validate = true }: { validate?: boolean } = {},
): Promise<void> {
  if (kernels.length === 0) return;
  const device: GPUDevice | undefined = renderer.backend?.device;
  if (validate && device) device.pushErrorScope('validation');
  renderer.compute(kernels.map((k) => k.node));
  if (validate && device) {
    const error = await device.popErrorScope();
    if (error) throw new Error(`compute pass rejected (first kernel "${kernels[0].label}"): ${error.message}`);
  }
}

/**
 * The WGSL three generated for a kernel (builds it if needed). For tests: snapshot drift checks and assertions such as
 * "no f32 comparison of indices" (R2).
 */
export function kernelWGSL(renderer: NRRendererLike, k: NRKernel): string {
  return renderer._nodes.getForCompute(k.node).computeShader as string;
}
