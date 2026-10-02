// Port of OpenDLSS-NR ports/browser-webgpu/src/model.js (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/src/model.js).
//
// Loading the weights. The model directory (docs/weights.md in the reference) is a manifest plus eleven stage files
// of packed bytes; a record is a slice of a stage. The reference uploads each stage whole and addresses FP8 matrices
// as byte offsets into it; this port instead copies every matrix into its own storage attribute, so kernels that
// differ only in which block's weights they read generate identical WGSL (design decision 0.3). The bytes stay in
// the model file's MMA fragment order (`packedWeightIndex`), exactly as the reference's GEMM reads them.
//
// What is rebuilt on the host is what no kernel addresses in place, byte for byte as the reference rebuilds it: the
// two f16 matrices (`f16Matrix`), the attention prior with its query axis untangled (`relativeBias`), the per-head f32
// scales, and the per-channel half vectors (`auxVector`, `auxPair`; the reference's GEMM reads its skip scales from
// the stage buffer instead, `auxOffset`, which here is an `auxVector` of the same bytes).

import { alignUp } from '../geometry.js';
import { f16ToNumber, inverseTiledToken, packedF16WeightIndex, tiledToken } from '../numerics/oracle.js';
import { attributeFromBytes, createF32Vector, createHalfVector } from '../tensors.js';
import type { F16Matrix, F32Vector, FP8Matrix, HalfVector } from '../types.js';
import { parseManifest, sha256Hex, validateManifest, type NRManifest } from './manifest.js';

/** One record of the model: a slice of a stage. */
export interface NRModelTensor {
  readonly name: string;
  readonly block: number;
  readonly layer: number;
  readonly stage: string;
  readonly stageOffset: number;
  readonly byteLength: number;
  /** The record's bytes (a view into its stage). */
  readonly bytes: Uint8Array;
}

/**
 * A model directory held in memory: `manifest.json` and `model/<stage.file>` keyed by their paths relative to the
 * directory (what `generateSyntheticModel` returns). `manifest` may be given parsed instead of as a file.
 */
export interface NRModelFiles {
  readonly manifest?: NRManifest;
  readonly files: ReadonlyMap<string, Uint8Array>;
}

/** A directory URL (`manifest.json` and `model/...` below it) or an in-memory model. */
export type NRModelSource = string | URL | NRModelFiles;

export interface NRModelLoadOptions {
  /** Check each stage against its manifest SHA-256 (default false; the reference does not verify). */
  verify?: boolean;
  /** Called with (loadedBytes, totalBytes) after each stage. */
  onProgress?: (loaded: number, total: number) => void;
  /** The fetch used for URL sources (default `globalThis.fetch`). */
  fetch?: typeof globalThis.fetch;
}

/** Options of `fp8Matrix`. */
export interface FP8MatrixOptions {
  /** Input channels of one batch of a batched matrix (default: all of `k`). */
  batchK?: number;
}

const readHalf = (bytes: Uint8Array, at: number): number => bytes[at] | (bytes[at + 1] << 8);

/** The bytes of one FP8 matrix (`k * n` bytes at `byteOffset`), after the reference's shape and bound checks. */
export function fp8MatrixBytes(
  tensor: NRModelTensor,
  byteOffset: number,
  k: number,
  n: number,
  batchK = 0,
): Uint8Array {
  const batch = batchK || k;
  if (k % 32 || n % 16 || batch % 32 || k % batch) {
    throw new Error(`FP8 matrix shape must be K%32==0, N%16==0, batchK | K: ${tensor.name}`);
  }
  if (byteOffset + k * n > tensor.byteLength) throw new Error(`FP8 matrix exceeds tensor ${tensor.name}`);
  return tensor.bytes.subarray(byteOffset, byteOffset + k * n);
}

/**
 * The reference's load-time weight bound (model.js:186-208): every weight satisfies |w| <= 9 (magnitude code
 * <= 0x51), or is the NaN code 0x7f/0xff, which the weight table decodes to zero. The bounded-half GEMM is exact only
 * under it, so a violation is an error. Returns the index of the first offending byte, or -1.
 */
export function firstUnboundedWeight(bytes: Uint8Array): number {
  for (let i = 0; i < bytes.length; ++i) {
    const magnitude = bytes[i] & 0x7f;
    if (magnitude > 0x51 && magnitude !== 0x7f) return i;
  }
  return -1;
}

/** A plain `[K][alignUp(N, 16)]` f16 matrix from the model's fragment order (`model.js` `f16Matrix`). */
export function relayoutF16Matrix(tensor: NRModelTensor, byteOffset: number, k: number, n: number): Uint16Array {
  if (k % 16) throw new Error('f16 matrix K must be a multiple of 16');
  const paddedN = alignUp(n, 16);
  const plain = new Uint16Array(k * paddedN);
  for (let row = 0; row < k; ++row) {
    for (let column = 0; column < n; ++column) {
      const halfIndex = (byteOffset >> 1) + packedF16WeightIndex(row, column, n);
      if (halfIndex * 2 + 1 >= tensor.byteLength) throw new Error(`f16 matrix exceeds tensor ${tensor.name}`);
      plain[row * paddedN + column] = readHalf(tensor.bytes, halfIndex * 2);
    }
  }
  return plain;
}

/**
 * The attention prior as halves `[heads][64 natural query][64 natural key]` (`model.js` `relativeBias`). The model
 * stores it as the C operand of the score MMA: 16x16 fragments with both token axes in 4x4-tiled order.
 */
export function relayoutRelativeBias(tensor: NRModelTensor, byteOffset: number, heads: number): Uint16Array {
  const prior = new Uint16Array(heads * 64 * 64);
  for (let head = 0; head < heads; ++head) {
    for (let query = 0; query < 64; ++query) {
      const q = tiledToken(query);
      for (let key = 0; key < 64; ++key) {
        const m = q & 15;
        const n = key & 15;
        const lane = ((m & 7) << 2) | ((n & 7) >> 1);
        const fragment = (m >= 8 ? 2 : 0) + (n & 1);
        const halfIndex = (q >> 4) * 1024 + (key >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment;
        const byteIndex = byteOffset + head * 8192 + halfIndex * 2;
        if (byteIndex + 1 >= tensor.byteLength) throw new Error(`relative bias exceeds tensor ${tensor.name}`);
        prior[(head * 64 + query) * 64 + inverseTiledToken(key)] = readHalf(tensor.bytes, byteIndex);
      }
    }
  }
  return prior;
}

/** `count` f32 values at `byteOffset` (the per-head attention scales). */
export function readF32s(tensor: NRModelTensor, byteOffset: number, count: number): Float32Array {
  if (byteOffset + count * 4 > tensor.byteLength) throw new Error(`f32 values exceed tensor ${tensor.name}`);
  const view = new DataView(tensor.bytes.buffer, tensor.bytes.byteOffset, tensor.bytes.byteLength);
  return Float32Array.from({ length: count }, (_, i) => view.getFloat32(byteOffset + i * 4, true));
}

/** `count` half bit patterns at `byteOffset` (a per-channel scale vector). */
export function readHalves(tensor: NRModelTensor, byteOffset: number, count: number): Uint16Array {
  if (byteOffset % 2) throw new Error(`halves of ${tensor.name} are not half-aligned`);
  if (byteOffset + count * 2 > tensor.byteLength) throw new Error(`halves exceed tensor ${tensor.name}`);
  return Uint16Array.from({ length: count }, (_, i) => readHalf(tensor.bytes, byteOffset + i * 2));
}

const directoryUrl = (source: string | URL): string => String(source).replace(/\/+$/, '');

async function fetchBytes(fetcher: typeof globalThis.fetch, url: string, what: string): Promise<Uint8Array> {
  const response = await fetcher(url);
  if (!response.ok) throw new Error(`cannot read ${what} (${url}: ${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * The network's weights. Every accessor returns a storage attribute built once per key (the reference's
 * `gpuBuffer` keys), so re-recording the graph reuses them.
 */
export class NRModel {
  readonly manifest: NRManifest;
  readonly blockCount: number;
  readonly tensors = new Map<string, NRModelTensor>();
  /** Stage bytes by stage id. */
  readonly stages = new Map<string, Uint8Array>();
  private readonly cache = new Map<string, FP8Matrix | F16Matrix | HalfVector | F32Vector>();
  private readonly checked = new Set<string>();

  /** Build from a manifest and its stage files (keyed by stage id); validates both. */
  constructor(manifest: NRManifest, stages: ReadonlyMap<string, Uint8Array>) {
    validateManifest(manifest);
    this.manifest = manifest;
    this.blockCount = manifest.totals.blockCount;
    for (const stage of manifest.stages) {
      const bytes = stages.get(stage.id);
      if (!bytes) throw new Error(`missing stage ${stage.id}`);
      if (bytes.byteLength !== stage.packedByteLength) throw new Error(`stage size mismatch: ${stage.id}`);
      this.stages.set(stage.id, bytes);
    }
    for (const entry of manifest.tensors) {
      const stage = this.stages.get(entry.stage)!;
      this.tensors.set(entry.name, {
        name: entry.name,
        block: entry.block,
        layer: entry.layer,
        stage: entry.stage,
        stageOffset: entry.stageOffset,
        byteLength: entry.byteLength,
        bytes: stage.subarray(entry.stageOffset, entry.stageOffset + entry.byteLength),
      });
    }
  }

  /**
   * Load a model directory: a URL (`<url>/manifest.json`, `<url>/model/<stage.file>`, as the reference fetches) or an
   * in-memory `NRModelFiles` (e.g. `generateSyntheticModel()`).
   */
  static async load(source: NRModelSource, options: NRModelLoadOptions = {}): Promise<NRModel> {
    const fetcher = options.fetch ?? globalThis.fetch;
    let manifest: NRManifest;
    let read: (path: string, what: string) => Promise<Uint8Array>;
    if (typeof source === 'string' || source instanceof URL) {
      const directory = directoryUrl(source);
      read = (path, what) => fetchBytes(fetcher, `${directory}/${path}`, what);
      manifest = parseManifest(JSON.parse(new TextDecoder().decode(await read('manifest.json', 'the manifest'))));
    } else {
      const files = source.files;
      read = async (path, what) => {
        const bytes = files.get(path);
        if (!bytes) throw new Error(`cannot read ${what} (${path} is missing)`);
        return bytes;
      };
      manifest =
        source.manifest ??
        parseManifest(JSON.parse(new TextDecoder().decode(await read('manifest.json', 'the manifest'))));
    }
    validateManifest(manifest);
    const total = manifest.stages.reduce((sum, stage) => sum + stage.packedByteLength, 0);
    let loaded = 0;
    const stages = new Map<string, Uint8Array>();
    for (const stage of manifest.stages) {
      const bytes = await read(`model/${stage.file}`, `stage ${stage.id}`);
      if (bytes.byteLength !== stage.packedByteLength) throw new Error(`stage size mismatch: ${stage.id}`);
      if (options.verify) {
        if (!stage.sha256) throw new Error(`stage ${stage.id} has no sha256 to verify`);
        const digest = await sha256Hex(bytes);
        if (digest !== stage.sha256.toLowerCase()) throw new Error(`stage ${stage.id} fails its sha256 check`);
      }
      stages.set(stage.id, bytes);
      loaded += bytes.byteLength;
      options.onProgress?.(loaded, total);
    }
    return new NRModel(manifest, stages);
  }

  /** The record `blockN.layerM.parameter`. */
  tensor(block: number, layer = 0, parameter = 'layer'): NRModelTensor {
    const name = `block${block}.layer${layer}.${parameter}`;
    const found = this.tensors.get(name);
    if (!found) throw new Error(`missing tensor ${name}`);
    return found;
  }

  private cached<T extends FP8Matrix | F16Matrix | HalfVector | F32Vector>(key: string, build: () => T): T {
    const existing = this.cache.get(key);
    if (existing) return existing as T;
    const built = build();
    this.cache.set(key, built);
    return built;
  }

  /**
   * One FP8 matrix (`model.js` `fp8Matrix`): `k * n` bytes at `byteOffset` in fragment order, copied into its own
   * attribute with one word of zero slack, after the shape check and the |w| <= 9 bound check.
   */
  fp8Matrix(
    tensor: NRModelTensor,
    byteOffset: number,
    k: number,
    n: number,
    { batchK = 0 }: FP8MatrixOptions = {},
  ): FP8Matrix {
    const batch = batchK || k;
    const key = `${tensor.name}/${byteOffset}/${k}x${n}`;
    const bytes = fp8MatrixBytes(tensor, byteOffset, k, n, batch);
    if (!this.checked.has(key)) {
      const bad = firstUnboundedWeight(bytes);
      if (bad >= 0) throw new Error(`weight ${bad} of ${tensor.name} is outside the bounded-half range`);
      this.checked.add(key);
    }
    return this.cached(`${key}/${batch}`, () => ({ attribute: attributeFromBytes(bytes, 4), k, n, batchK: batch }));
  }

  /** A plain `[k][paddedN]` f16 matrix: the input adapter (16 -> 32) and the head (32 -> 4). */
  f16Matrix(tensor: NRModelTensor, byteOffset: number, k: number, n: number): F16Matrix {
    return this.cached(`${tensor.name}/f16/${byteOffset}/${k}x${n}`, () => ({
      attribute: attributeFromBytes(relayoutF16Matrix(tensor, byteOffset, k, n)),
      k,
      n,
      paddedN: alignUp(n, 16),
    }));
  }

  /** The window attention prior, halves `[heads][64 natural query][64 natural key]`. */
  relativeBias(tensor: NRModelTensor, byteOffset: number, heads: number): HalfVector {
    return this.cached(`${tensor.name}/prior/${byteOffset}/${heads}`, () =>
      createHalfVector(relayoutRelativeBias(tensor, byteOffset, heads)),
    );
  }

  /** The per-head f32 attention scales. */
  headScales(tensor: NRModelTensor, byteOffset: number, heads: number): F32Vector {
    return this.cached(`${tensor.name}/heads/${byteOffset}/${heads}`, () =>
      createF32Vector(readF32s(tensor, byteOffset, heads)),
    );
  }

  /** Per-channel skip scales (halves), indexed by output column. Also what the GEMM's `scale` wants. */
  auxVector(tensor: NRModelTensor, byteOffset: number, count: number): HalfVector {
    return this.cached(`${tensor.name}/aux/${byteOffset}/${count}`, () =>
      createHalfVector(readHalves(tensor, byteOffset, count)),
    );
  }

  /** Two scale vectors end to end (the post blend's input and adapter scales). */
  auxPair(tensor: NRModelTensor, offsetA: number, offsetB: number, count: number): HalfVector {
    return this.cached(`${tensor.name}/auxpair/${offsetA}/${offsetB}/${count}`, () => {
      const values = new Uint16Array(count * 2);
      values.set(readHalves(tensor, offsetA, count), 0);
      values.set(readHalves(tensor, offsetB, count), count);
      return createHalfVector(values);
    });
  }

  /** One f16 value as a half bit pattern. */
  auxHalf(tensor: NRModelTensor, byteOffset: number, column: number): number {
    return readHalf(tensor.bytes, byteOffset + column * 2);
  }

  /** One f32 value. */
  auxF32(tensor: NRModelTensor, byteOffset: number): number {
    return readF32s(tensor, byteOffset, 1)[0];
  }

  /** The learned temporal blend scale (`block70.layer0.blend_scale`, one half). */
  blendScale(): number {
    const tensor = this.tensor(70, 0, 'blend_scale');
    if (tensor.byteLength < 2) throw new Error('the model has no blend scale');
    return f16ToNumber(readHalf(tensor.bytes, 0));
  }

  /** Forget every built attribute (three releases their GPU buffers when they are no longer referenced). */
  dispose(): void {
    this.cache.clear();
    this.checked.clear();
  }
}
