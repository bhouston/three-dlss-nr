// The reference `Model` (OpenDLSS-NR's WebGPU port, by maan, MIT; ports/browser-webgpu/src/model.js) and a
// record-only pass of its `Graph` (src/graph.js), for comparing our `NRModel` with it buffer by buffer.
//
// Part of three-dlss-nr. `recordGraphRequests` runs the reference `Graph.record` with stub pipelines and a recorder
// that only counts, so it needs no GPU and no `shader-f16`; what it does exercise is every weight request the graph
// makes (and the checks the reference makes on them: the |w| <= 9 bound in `fp8Matrix`, the layout lengths in
// `record`). Each request is logged with the reference's result, so a test can rebuild the same buffer with
// `NRModel` and compare bytes.

import { geometryFromValid } from '@ref/geometry.js';
import { Graph } from '@ref/graph.js';
import { Model } from '@ref/model.js';

/** A `GPUDevice` stand-in that keeps what is written to each buffer (the reference `Model` needs nothing else). */
export interface FakeBuffer {
  label: string;
  size: number;
  data: Uint8Array;
  destroy(): void;
}

export function fakeDevice(): { device: any; buffers: FakeBuffer[] } {
  const buffers: FakeBuffer[] = [];
  const device = {
    createBuffer({ label, size }: { label: string; size: number }): FakeBuffer {
      const buffer = { label, size, data: new Uint8Array(size), destroy() {} };
      buffers.push(buffer);
      return buffer;
    },
    queue: {
      writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBufferView) {
        buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
    },
  };
  // The reference reads the usage flags from the WebGPU globals, which the unit project does not have.
  const globals = globalThis as any;
  globals.GPUBufferUsage ??= { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
  return { device, buffers };
}

/** A device wrapper whose buffers can also be copied from (the reference `Model` creates STORAGE | COPY_DST only). */
export function readableDevice(device: GPUDevice): any {
  return {
    createBuffer: (descriptor: GPUBufferDescriptor) =>
      device.createBuffer({ ...descriptor, usage: descriptor.usage | (globalThis as any).GPUBufferUsage.COPY_SRC }),
    queue: device.queue,
  };
}

/** Load the reference `Model` from a directory URL (e.g. `synthetic://nr` registered with the fetch shim). */
export async function loadReferenceModelOn(device: any, directory: string): Promise<any> {
  return new Model(device).load(directory);
}

/** One weight request of the reference graph: the `Model` method, its arguments, and what it returned. */
export interface ModelRequest {
  method: 'fp8Matrix' | 'f16Matrix' | 'relativeBias' | 'headScales' | 'auxVector' | 'auxPair' | 'auxOffset';
  tensor: string;
  args: any[];
  result: any;
}

const LOGGED: ModelRequest['method'][] = [
  'fp8Matrix',
  'f16Matrix',
  'relativeBias',
  'headScales',
  'auxVector',
  'auxPair',
  'auxOffset',
];

/** One dispatch of the reference graph as its recorder sees it. */
export interface RecordedPass {
  label: string;
  /** The entry point (`Recorder.pass`) or the specialized kernel's name (`gemm_fp8`, `window_attend`). */
  kind: string;
  /** Workgroup counts `[x, y, z]`. */
  dispatch: [number, number, number];
}

const dim3 = (groups: number | number[]): [number, number, number] => {
  const [x, y = 1, z = 1] = Array.isArray(groups) ? groups : [groups];
  return [x, y, z];
};

/**
 * Record the reference graph for a valid size against `model` (a loaded reference `Model`) without a GPU pipeline:
 * returns every weight request in order, the dispatch labels, every dispatch with its kind and size, and the
 * boundary captures in order (`capture <name>` after the index of the dispatch they follow).
 */
export function recordGraphRequests(
  model: any,
  width: number,
  height: number,
  {
    captureBoundaries = false,
    onAllocate,
  }: {
    captureBoundaries?: boolean;
    /** Called for every tensor the graph allocates (`Tensors.allocate` arguments). */
    onAllocate?: (label: string, rows: number, channels: number, format: string) => void;
  } = {},
): { requests: ModelRequest[]; dispatches: string[]; passes: RecordedPass[]; captures: string[] } {
  const requests: ModelRequest[] = [];
  const logged = Object.create(model);
  for (const method of LOGGED) {
    logged[method] = (tensor: any, ...args: any[]) => {
      const result = model[method](tensor, ...args);
      requests.push({ method, tensor: tensor.name, args, result });
      return result;
    };
  }
  const dispatches: string[] = [];
  const passes: RecordedPass[] = [];
  const captures: string[] = [];
  const recorder = {
    pass: (entry: string, _buffers: unknown, _params: unknown, groups: number | number[], label: string) => {
      dispatches.push(label);
      passes.push({ label, kind: entry, dispatch: dim3(groups) });
    },
    specialized: (
      _p: unknown,
      { kernel }: { kernel: string },
      _b: unknown,
      _params: unknown,
      groups: number[],
      label: string,
    ) => {
      dispatches.push(label);
      passes.push({ label, kind: kernel, dispatch: dim3(groups) });
    },
    copy: (_from: unknown, _to: unknown, _bytes: number, label: string) => {
      captures.push(`${dispatches.length - 1} ${label}`);
    },
  };
  const stub = {};
  const geometry = geometryFromValid(width, height);
  const graph = new Graph(
    {
      device: null,
      kernels: { pipelineLayout: stub, gemmPipelineLayout: stub },
      matmul: { pipeline: () => stub, siluTableFor: () => stub, weightMetadata: stub },
      window: { pipeline: () => stub },
      tensors: {
        allocate: (label: string, rows: number, channels: number, format: string) => {
          onAllocate?.(label, rows, channels, format);
          return {
            label,
            rows,
            channels,
            format,
            buffer: stub,
          };
        },
      },
      model: logged,
      geometry,
    },
    { captureBoundaries },
  );
  graph.record(recorder, {
    label: 'input features',
    rows: geometry.fullRows,
    channels: 16,
    format: 'f32',
    buffer: stub,
  });
  return { requests, dispatches, passes, captures };
}

/** Reads a reference buffer's bytes (a `FakeBuffer`'s data, or a GPU read-back). */
export type BufferReader = (buffer: any) => Promise<Uint8Array>;

/** Reads `FakeBuffer`s. */
export const fakeReader: BufferReader = async (buffer: FakeBuffer) => buffer.data;

const sameBytes = (a: Uint8Array, b: Uint8Array): boolean =>
  a.length === b.length && a.every((byte, i) => byte === b[i]);

const attributeBytes = (attribute: { array: Uint32Array }) =>
  new Uint8Array(attribute.array.buffer, attribute.array.byteOffset, attribute.array.byteLength);

/**
 * Rebuild every request with `ours` (an `NRModel`) and compare bytes with the reference's buffer. Returns the
 * mismatches as messages (empty when every buffer is byte-identical) and how many requests were compared.
 */
export async function compareRequests(
  requests: ModelRequest[],
  refModel: any,
  ours: any,
  read: BufferReader,
): Promise<{ compared: number; mismatches: string[] }> {
  const mismatches: string[] = [];
  const stageBytes = new Map<any, Promise<Uint8Array>>();
  const readStage = (buffer: any) => {
    if (!stageBytes.has(buffer)) stageBytes.set(buffer, read(buffer));
    return stageBytes.get(buffer)!;
  };
  let compared = 0;
  for (const request of requests) {
    const tensor = ours.tensors.get(request.tensor);
    const what = `${request.method}(${request.tensor}, ${JSON.stringify(request.args)})`;
    let expected: Uint8Array;
    let actual: Uint8Array;
    switch (request.method) {
      case 'fp8Matrix': {
        const [byteOffset, k, n, options] = request.args;
        const { buffer, byteOffset: stageOffset, batchK } = request.result;
        expected = (await readStage(buffer)).slice(stageOffset, stageOffset + k * n);
        const matrix = ours.fp8Matrix(tensor, byteOffset, k, n, options);
        if (matrix.k !== k || matrix.n !== n || matrix.batchK !== batchK) mismatches.push(`${what}: shape`);
        const bytes = attributeBytes(matrix.attribute);
        if (bytes.length < k * n + 4 || bytes.subarray(k * n).some((byte) => byte !== 0)) {
          mismatches.push(`${what}: slack is not zero`);
        }
        actual = bytes.slice(0, k * n);
        break;
      }
      case 'auxOffset': {
        const [byteOffset, count] = request.args;
        const stage = refModel.stage(refModel.tensors.get(request.tensor));
        expected = (await readStage(stage)).slice(request.result, request.result + count * 2);
        actual = attributeBytes(ours.auxVector(tensor, byteOffset, count).attribute).slice(0, count * 2);
        break;
      }
      case 'f16Matrix': {
        const [byteOffset, k, n] = request.args;
        expected = await read(request.result.buffer);
        const matrix = ours.f16Matrix(tensor, byteOffset, k, n);
        if (matrix.paddedN !== request.result.paddedN) mismatches.push(`${what}: paddedN`);
        actual = attributeBytes(matrix.attribute);
        break;
      }
      default: {
        expected = await read(request.result);
        actual = attributeBytes((ours[request.method] as (...a: any[]) => any)(tensor, ...request.args).attribute);
      }
    }
    compared += 1;
    if (!sameBytes(actual, expected)) {
      const first = actual.findIndex((byte, i) => byte !== expected[i]);
      mismatches.push(`${what}: ${actual.length} vs ${expected.length} bytes, first difference at ${first}`);
    }
  }
  return { compared, mismatches };
}
