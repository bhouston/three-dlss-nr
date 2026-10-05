// Exercise the actual pinned graph and compiler against a recording WebGPU boundary.
// No shaders execute here; output parity belongs to the GPU tests.
import { beforeAll, afterAll, expect, it, vi } from 'vitest';
import { generateSyntheticModel, type SyntheticModel } from '../synthetic/generate.js';
import { loadReferenceModel, ReferenceWgslBackend } from './ReferenceWgslBackend.js';

let synthetic: SyntheticModel;
beforeAll(async () => {
  vi.stubGlobal('GPUBufferUsage', { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 4 });
  synthetic = await generateSyntheticModel();
});
afterAll(() => vi.unstubAllGlobals());

function recordingDevice() {
  const descriptors: GPUComputePipelineDescriptor[] = [];
  const buffers: { destroyed: boolean; size: number; destroy(): void }[] = [];
  let failEntry: string | undefined;
  let lose!: () => void;
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device = {
    features: new Set(['shader-f16']),
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeInvocationsPerWorkgroup: 512,
    },
    lost: new Promise<void>((resolve) => {
      lose = resolve;
    }),
    queue: { writeBuffer() {}, submit() {}, async onSubmittedWorkDone() {} },
    createBuffer({ size }: { size: number }) {
      const buffer = {
        size,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroupLayout(descriptor: unknown) {
      return { descriptor };
    },
    createPipelineLayout(descriptor: unknown) {
      return { descriptor };
    },
    createBindGroup(descriptor: unknown) {
      return { descriptor };
    },
    createShaderModule(descriptor: unknown) {
      return {
        descriptor,
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    async createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor) {
      descriptors.push(descriptor);
      if (descriptor.compute.entryPoint === failEntry) {
        failEntry = undefined;
        throw new Error('injected preparation failure');
      }
      return {
        descriptor,
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return pass;
        },
        finish() {
          return {};
        },
      };
    },
  } as unknown as GPUDevice;
  const attributes = new Map();
  const renderer = {
    backend: {
      device,
      createStorageAttribute(attribute: any) {
        if (!attributes.has(attribute))
          attributes.set(attribute, { buffer: device.createBuffer({ size: attribute.array.byteLength, usage: 128 }) });
      },
      get(attribute: unknown) {
        return attributes.get(attribute);
      },
      destroyAttribute(attribute: unknown) {
        attributes.get(attribute).buffer.destroy();
        attributes.delete(attribute);
      },
    },
  };
  return {
    device,
    renderer,
    descriptors,
    buffers,
    lose,
    fail(entry: string) {
      failEntry = entry;
    },
  };
}

async function fixture() {
  const recording = recordingDevice();
  const model = await loadReferenceModel(recording.renderer, synthetic);
  return {
    ...recording,
    model,
    create(width = 64, extraShaders?: Parameters<typeof ReferenceWgslBackend.create>[0]['extraShaders']) {
      return ReferenceWgslBackend.create({ renderer: recording.renderer, model, width, height: 64, extraShaders });
    },
  };
}

it('compiles cold and new shapes, then reuses every pipeline on A → B → A', async () => {
  const f = await fixture();
  const first = await f.create();
  const coldCount = f.descriptors.length;
  expect(coldCount).toBeGreaterThan(20);
  first.dispose();
  const resized = await f.create(128);
  expect(f.descriptors.length).toBeGreaterThan(coldCount);
  resized.dispose();
  const warmCount = f.descriptors.length;
  const again = await f.create();
  expect(f.descriptors).toHaveLength(warmCount);
  expect(again.dispatchCount).toBe(first.dispatchCount);
  again.dispose();
  f.model.dispose();
});

it('isolates concurrent graph buffers and survives disposing a sibling graph', async () => {
  const f = await fixture();
  const [a, b] = await Promise.all([f.create(), f.create()]);
  expect(a.network.kernels).toBe(b.network.kernels);
  expect(a.network.matmul).toBe(b.network.matmul);
  expect(a.network.recorder).not.toBe(b.network.recorder);
  expect(a.network.tensors).not.toBe(b.network.tensors);
  expect(a.network.features.buffer).not.toBe(b.network.features.buffer);
  expect(a.network.graph.head.buffer).not.toBe(b.network.graph.head.buffer);
  const siblingBuffers = [...b.network.tensors.byKey.values()].map((tensor: any) => tensor.buffer);
  a.dispose();
  expect(siblingBuffers.every((buffer) => !buffer.destroyed)).toBe(true);
  b.network.validated = true;
  await b.run();
  const count = f.descriptors.length;
  const c = await f.create();
  expect(f.descriptors).toHaveLength(count);
  b.dispose();
  c.dispose();
  f.model.dispose();
});

it('keeps devices, token padding, shader source and entry point lists separate', async () => {
  const f = await fixture();
  const a = await f.create();
  const padded = await f.create(1024);
  expect(padded.network.geometry.paddedVitTokens).not.toBe(a.network.geometry.paddedVitTokens);
  expect(padded.network.kernels).not.toBe(a.network.kernels);
  const shader = { name: 'extra', code: '@compute @workgroup_size(1) fn extra() {}', entryPoints: ['extra'] };
  const extra = await f.create(64, [shader]);
  const same = await f.create(64, [{ ...shader, entryPoints: [...shader.entryPoints] }]);
  expect(same.network.kernels).toBe(extra.network.kernels);
  const changedCode = await f.create(64, [{ ...shader, code: shader.code + '\n// changed' }]);
  const changedEntries = await f.create(64, [{ ...shader, entryPoints: ['extra', 'other'] }]);
  expect(changedCode.network.kernels).not.toBe(extra.network.kernels);
  expect(changedEntries.network.kernels).not.toBe(extra.network.kernels);
  const changedName = await f.create(64, [{ ...shader, name: 'other-name' }]);
  expect(changedName.network.kernels).not.toBe(extra.network.kernels);
  const mutable = { name: 'snapshot', code: shader.code, entryPoints: ['snapshot'] };
  const preparing = f.create(64, [mutable]);
  mutable.entryPoints[0] = 'mutated';
  mutable.code += '// mutated';
  const snapshot = await preparing;
  expect(snapshot.network.kernels.pipelines.has('snapshot')).toBe(true);
  expect(snapshot.network.kernels.pipelines.has('mutated')).toBe(false);

  const other = await fixture();
  const otherGraph = await other.create();
  expect(otherGraph.network.kernels).not.toBe(a.network.kernels);
  for (const backend of [a, padded, extra, same, changedCode, changedEntries, changedName, snapshot, otherGraph])
    backend.dispose();
  f.model.dispose();
  other.model.dispose();
});

it('retries rejected preparation and drops program identity after device loss', async () => {
  const f = await fixture();
  f.fail('gemm_f16');
  await expect(f.create()).rejects.toThrow('injected preparation failure');
  const backend = await f.create();
  f.lose();
  await Promise.resolve();
  // Real lost devices cannot create a graph. Reusing this recording object probes cache removal only.
  const afterLoss = await f.create();
  expect(afterLoss.network.kernels).not.toBe(backend.network.kernels);
  backend.dispose();
  afterLoss.dispose();
  f.model.dispose();
});

it('keeps specialization constants in the pinned compiler cache key', async () => {
  const f = await fixture();
  const a = await f.create();
  const matmul = a.network.matmul;
  const layout = a.network.kernels.gemmPipelineLayout;
  const variant = { output: 'half', residual: 'e4', batched: true, tile128: true };
  const shape = {
    flags: 0,
    rows: 64,
    k: 32,
    n: 32,
    weightByteOffset: 0,
    biasByteOffset: 0,
    weightMatrixChannels: 32,
    outputMatrixChannels: 32,
    inputMatrixChannels: 32,
  };
  const gemm = await matmul.pipeline(layout, variant, shape);
  expect(await matmul.pipeline(layout, { ...variant }, { ...shape })).toBe(gemm);
  for (const field of Object.keys(shape) as (keyof typeof shape)[]) {
    expect(await matmul.pipeline(layout, variant, { ...shape, [field]: shape[field] + 32 })).not.toBe(gemm);
  }
  expect(await matmul.pipeline(layout, { ...variant, output: 'e4' }, shape)).not.toBe(gemm);
  f.fail('main');
  const retryShape = { ...shape, rows: 192 };
  await expect(matmul.pipeline(layout, variant, retryShape)).rejects.toThrow('injected preparation failure');
  await expect(matmul.pipeline(layout, variant, retryShape)).resolves.toBeDefined();

  // A changed window shift is a simple direct probe of the actual upstream compiler.
  const window = a.network.window;
  const windowShape = { width: 8, height: 8, channels: 32, shiftX: 0, shiftY: 0, relativeBias: true };
  const p = await window.pipeline(a.network.kernels.pipelineLayout, windowShape);
  expect(await window.pipeline(a.network.kernels.pipelineLayout, { ...windowShape })).toBe(p);
  expect(await window.pipeline(a.network.kernels.pipelineLayout, { ...windowShape, shiftX: 4 })).not.toBe(p);
  a.dispose();
  f.model.dispose();
});
