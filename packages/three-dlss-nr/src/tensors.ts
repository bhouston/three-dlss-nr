// Activation tensors and small weight vectors as three.js storage attributes.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). `NRTensors` mirrors the reference's
// ports/browser-webgpu/src/passes.js `Tensors`: rows padded to a multiple of 64, the whole allocation zero-filled
// (a kernel reading a window or key block at the end of a tensor reads past the last valid row, and what it reads
// there has to be zero), keyed by `label/rows x channels/format` so the graph reuses one buffer per key.
//
// Every buffer is a `StorageBufferAttribute` over a `Uint32Array`. Sharing a tensor between kernels needs nothing:
// one attribute object is one GPUBuffer.

import { StorageBufferAttribute as ThreeStorageBufferAttribute } from 'three/webgpu';

import { alignUp } from './geometry.js';
import type { F32Vector, HalfVector, NRTensor, StorageBufferAttribute, TensorFormat } from './types.js';

/** Bytes per value of a tensor format. */
export const bytesPerValue = (format: TensorFormat): number => (format === 'f32' ? 4 : format === 'f16' ? 2 : 1);

/** A zero-filled storage attribute of at least `byteLength` bytes (rounded up to whole words, at least one). */
export function wordAttribute(byteLength: number): StorageBufferAttribute {
  return new ThreeStorageBufferAttribute(new Uint32Array(Math.max(1, Math.ceil(byteLength / 4))), 1);
}

/** A storage attribute holding a copy of `bytes` (zero-padded to whole words, plus `slackBytes` of zeros). */
export function attributeFromBytes(bytes: ArrayBufferView, slackBytes = 0): StorageBufferAttribute {
  const attribute = wordAttribute(bytes.byteLength + slackBytes);
  new Uint8Array(attribute.array.buffer).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return attribute;
}

/** A standalone activation tensor (not keyed), laid out exactly as the reference allocates one. */
export function createTensor(label: string, rows: number, channels: number, format: TensorFormat): NRTensor {
  const allocRows = alignUp(rows, 64);
  const bpv = bytesPerValue(format);
  const byteLength = alignUp(allocRows * channels * bpv, 4);
  return {
    label,
    rows,
    channels,
    format,
    allocRows,
    attribute: wordAttribute(byteLength),
    byteLength,
    validBytes: rows * channels * bpv,
  };
}

/** The graph's tensor pool (`passes.js` `Tensors`). */
export class NRTensors {
  readonly byKey = new Map<string, NRTensor>();
  /** Bytes allocated so far. */
  total = 0;

  /** The tensor for `label/rows x channels/format`, allocated on first request. */
  allocate(label: string, rows: number, channels: number, format: TensorFormat): NRTensor {
    const key = `${label}/${rows}x${channels}/${format}`;
    const existing = this.byKey.get(key);
    if (existing) return existing;
    const tensor = createTensor(label, rows, channels, format);
    this.total += tensor.byteLength;
    this.byKey.set(key, tensor);
    return tensor;
  }

  /** Every tensor allocated under `label` (any shape). */
  byLabel(label: string): NRTensor[] {
    return [...this.byKey.values()].filter((tensor) => tensor.label === label);
  }

  /** Drop the pool. GPU buffers are released by three when their attributes are garbage collected or disposed. */
  clear(): void {
    this.byKey.clear();
    this.total = 0;
  }
}

/** A half vector from half bit patterns (two per word, element `2i` in the low 16 bits). */
export function createHalfVector(halves: Uint16Array): HalfVector {
  return { attribute: attributeFromBytes(halves), count: halves.length };
}

/** An f32 vector (bit patterns, one per word). */
export function createF32Vector(values: Float32Array): F32Vector {
  return { attribute: attributeFromBytes(values), count: values.length };
}

/** The bytes of a storage attribute's CPU copy. */
export const attributeBytes = (attribute: StorageBufferAttribute): Uint8Array =>
  new Uint8Array(attribute.array.buffer, attribute.array.byteOffset, attribute.array.byteLength);

/**
 * Overwrite bytes of a buffer from the CPU and schedule the upload (three re-uploads an attribute whose version
 * changed the next time a kernel binding it runs). Note the CPU copy does not track what kernels wrote on the GPU.
 */
export function writeBuffer(
  source: { attribute: StorageBufferAttribute },
  data: ArrayBufferView,
  byteOffset = 0,
): void {
  const bytes = attributeBytes(source.attribute);
  if (byteOffset + data.byteLength > bytes.byteLength) throw new RangeError('writeBuffer: data exceeds the buffer');
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), byteOffset);
  source.attribute.needsUpdate = true;
}

/** Fill a whole buffer with one byte value (tests prefill outputs with the sentinel 0xCD, R6). */
export function fillBuffer(source: { attribute: StorageBufferAttribute }, byte: number): void {
  attributeBytes(source.attribute).fill(byte);
  source.attribute.needsUpdate = true;
}

/**
 * Read a buffer back from the GPU (`renderer.getArrayBufferAsync`). For a tensor, `validOnly` (default) returns the
 * `validBytes` covering its logical rows, as the reference's readback does.
 */
export async function readBuffer(
  renderer: any,
  source: { attribute: StorageBufferAttribute; validBytes?: number },
  { validOnly = true }: { validOnly?: boolean } = {},
): Promise<Uint8Array> {
  const buffer: ArrayBuffer = await renderer.getArrayBufferAsync(source.attribute);
  const bytes = new Uint8Array(buffer);
  return validOnly && source.validBytes !== undefined ? bytes.slice(0, source.validBytes) : bytes;
}
