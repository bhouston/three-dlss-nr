// Shared types of the TSL port of the OpenDLSS-NR network.
//
// three-dlss-nr is a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan
// (https://github.com/maanHimself/OpenDLSS-NR, MIT, pinned at 9d08f41). The tensor and weight descriptors below
// mirror the objects the reference WebGPU port passes around (ports/browser-webgpu/src/passes.js `Tensors`,
// src/model.js `fp8Matrix` / `f16Matrix` / `auxVector`), re-expressed over three.js `StorageBufferAttribute`s.
//
// These interfaces are the contract between the chunks of the port (model, GEMMs, attention, ops, graph). Change
// them only together with every consumer.

/**
 * The parts of three.js' `StorageBufferAttribute` (from 'three/webgpu') the port relies on. three ships no TypeScript
 * declarations, so this structural type stands in for it. Every port buffer holds a `Uint32Array`.
 */
export interface StorageBufferAttribute {
  readonly isStorageBufferAttribute: true;
  readonly array: Uint32Array;
  /** Words (itemSize 1). */
  readonly count: number;
  /** Setting it schedules a re-upload of `array`. */
  needsUpdate: boolean;
}

/** three.js `ComputeNode` (from 'three/tsl' `.compute()`); untyped because three ships no declarations. */
export type ComputeNode = any;

/** Storage format of an activation tensor: E4M3FN bytes, IEEE half bit patterns, or f32. */
export type TensorFormat = 'e4' | 'f16' | 'f32';

/**
 * An activation tensor `[rows][channels]`, as the reference allocates it (`passes.js` `Tensors.allocate`): rows
 * padded to a multiple of 64 (`allocRows`), the whole allocation zero-filled, values packed little-endian into u32
 * words (four E4 bytes or two halves per word; f32 tensors hold the f32 bit patterns). Kernels view it as
 * `storage(attribute, 'uint')`.
 */
export interface NRTensor {
  /** Label without shape, e.g. `'block0 ffn'`; tensors are keyed by `label/rows x channels/format`. */
  readonly label: string;
  /** Logical rows (tokens / pixels). */
  readonly rows: number;
  /** Values per row; also the row stride in values. */
  readonly channels: number;
  readonly format: TensorFormat;
  /** `alignUp(rows, 64)`: rows actually allocated (the padding rows stay zero unless a kernel writes them). */
  readonly allocRows: number;
  /** The GPU buffer: a `StorageBufferAttribute` over a `Uint32Array` of `byteLength / 4` words. */
  readonly attribute: StorageBufferAttribute;
  /** Allocated bytes: `alignUp(allocRows * channels * bytesPerValue, 4)`. */
  readonly byteLength: number;
  /** Bytes covering the logical rows: `rows * channels * bytesPerValue` (what the reference reads back). */
  readonly validBytes: number;
}

/**
 * One FP8 (E4M3FN) weight matrix in the model file's MMA fragment order (`packedWeightIndex`), copied into its own
 * attribute (one word of slack past the end, as `model.js` pads stages). `k` counts all batches: a batched matrix
 * holds `k / batchK` matrices of `batchK x n` back to back (byte `batch * batchK * n + packedWeightIndex(k, n, n)`).
 */
export interface FP8Matrix {
  readonly attribute: StorageBufferAttribute;
  /** Total input channels (`batchK * batches`). */
  readonly k: number;
  /** Output channels of the matrix (the reference's `matrixChannels`). */
  readonly n: number;
  /** Input channels of one batch (the reference's `batchK`; equals `k` when not batched). */
  readonly batchK: number;
}

/** A vector of IEEE halves, two per u32 word (little-endian: element `2i` in the low 16 bits). */
export interface HalfVector {
  readonly attribute: StorageBufferAttribute;
  readonly count: number;
}

/** A vector of f32 values stored as their bit patterns, one per u32 word. */
export interface F32Vector {
  readonly attribute: StorageBufferAttribute;
  readonly count: number;
}

/**
 * A plain f16 matrix `[k][paddedN]` (halves, two per word), as `model.js` `f16Matrix` rebuilds the input adapter
 * and the head; `paddedN = alignUp(n, 16)`.
 */
export interface F16Matrix {
  readonly attribute: StorageBufferAttribute;
  readonly k: number;
  readonly n: number;
  readonly paddedN: number;
}

/** A 3D dispatch size or workgroup size. */
export type Dim3 = readonly [number, number, number];

/**
 * One dispatch of the graph: a TSL compute node with its own `dispatchSize`, plus the metadata the graph and the
 * tests need. Built by `kernel()` (`tsl/KernelBuilder.ts`).
 */
export interface NRKernel {
  /** The reference's dispatch label (e.g. `'block 5 expert expand'`), used for bisection and debugging. */
  readonly label: string;
  /** Kernel family, the reference's entry point name: `'gemm_fp8'`, `'gemm_f16'`, `'window_attend'`, ... */
  readonly kind: string;
  /** The three.js compute node; `renderer.compute([...nodes])` runs a list of them in one compute pass. */
  readonly node: ComputeNode;
  /** Tensors the kernel writes (R6: tests prefill these with a sentinel and compare them whole). */
  readonly writes: readonly NRTensor[];
  /** Tensors the kernel reads (bound read-only). */
  readonly reads: readonly NRTensor[];
  /** Workgroup counts `[x, y, z]`, as the reference dispatches them. */
  readonly dispatch: Dim3;
  readonly workgroupSize: Dim3;
}

/** One FP8 GEMM as the graph calls it (the reference's `Graph.gemm` arguments that are not buffers). */
export interface GemmSpec {
  /** Activation rows (not padded). */
  rows: number;
  /** Input channels of one matrix (`batchK`). */
  k: number;
  /** Output channels of one matrix. */
  n: number;
  /** Independent matrices sharing the dispatch (expert FFN); 1 otherwise. */
  batches: number;
  /** Every batch reads the same K input columns (expert expand). */
  broadcast: boolean;
  /** K span whose sums are published to half and added separately; 0 = none. */
  partition: 0 | 256 | 512 | 1024;
  /** Publish through the SiLU table (E4 output only). */
  silu: boolean;
  /** `e4` packed bytes, `half` raw halves, or `dual` (both, same index). */
  output: 'e4' | 'half' | 'dual';
  /** The skip tensor the accumulator is seeded from (scaled per column), and its storage format. */
  residual: null | { format: 'e4' | 'f16' };
  /** The reference's dispatch label. */
  label: string;
}

/** Buffers of one FP8 GEMM. */
export interface GemmFp8Buffers {
  input: NRTensor;
  weights: FP8Matrix;
  /** E4 output (`output: 'e4' | 'dual'`). */
  output?: NRTensor;
  /** Half output (`output: 'half' | 'dual'`). */
  outputF16?: NRTensor;
  /** Skip tensor (`residual !== null`). */
  residual?: NRTensor;
  /** Per-output-column skip scales (`residual !== null`). */
  scale?: HalfVector;
}

/** One f16 GEMM (input adapter 16 -> 32 and head 32 -> 4). */
export interface GemmF16Spec {
  rows: number;
  k: number;
  n: number;
  label: string;
}

export interface GemmF16Buffers {
  input: NRTensor;
  weights: F16Matrix;
  output?: NRTensor;
  outputF16?: NRTensor;
  outputF32?: NRTensor;
}

/** One shifted-window attention (`attend_window_tiled`). `phase` indexes `windowPhase` (0..3). */
export interface WindowAttentionSpec {
  width: number;
  height: number;
  heads: number;
  phase: number;
  /** Block label; the dispatch label is `${label} attend`, as in the reference. */
  label: string;
}

export interface WindowAttentionBuffers {
  /** Raw qkv halves `[rows][3 * heads * 32]`. */
  qkv: NRTensor;
  /** Prior `[heads][64 natural query][64 natural key]` halves (`model.js` `relativeBias`). */
  prior: HalfVector;
  /** Per-head f32 attention scales. */
  scales: F32Vector;
  /** E4 output `[rows][heads * 32]`. */
  attended: NRTensor;
}

/** The global ViT attention (`vit_normalize`, `vit_attend`); `paddedTokens = (tokens + 63) & ~63` is baked. */
export interface VitSpec {
  tokens: number;
  heads: number;
  paddedTokens: number;
  label: string;
}

export interface VitNormalizeBuffers {
  /** Raw qkv halves `[tokens][3 * 1024]`. */
  qkv: NRTensor;
  /** Per-head learned f32 scales. */
  scales: F32Vector;
  /** E4 `[paddedTokens][3 * 1024]`; padding rows are never written and must stay zero. */
  normalized: NRTensor;
}

export interface VitAttendBuffers {
  normalized: NRTensor;
  attended: NRTensor;
}
