// Run the real reference graph and compiler scheduler against a recording GPU boundary.
// This checks actual pipeline creation calls, not elapsed time or the progress counter.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import { BUFFER_USAGE } from '../backend/gpuFlags.js';
import { generateSyntheticModel, type SyntheticModel } from '../synthetic/generate.js';
import { loadReferenceModel, ReferenceWgslBackend } from './ReferenceWgslBackend.js';

let source: SyntheticModel;
beforeAll(async () => {
  source = await generateSyntheticModel({ seed: 1 });
});
afterEach(() => vi.unstubAllGlobals());

function recordingRenderer() {
  vi.stubGlobal('GPUBufferUsage', BUFFER_USAGE);
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 4 });
  const createComputePipelineAsync = vi.fn(async (_descriptor: GPUComputePipelineDescriptor) => ({
    getBindGroupLayout: () => ({}),
  }));
  const device = {
    features: new Set(['shader-f16']),
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeInvocationsPerWorkgroup: 512,
    },
    lost: new Promise(() => {}),
    createBuffer: ({ size, label }: GPUBufferDescriptor) => ({ size, label, destroy() {} }),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBindGroup: () => ({}),
    createComputePipelineAsync,
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      finish: () => ({}),
    }),
    queue: { writeBuffer() {}, submit() {} },
  };
  const attributes = new Map();
  const renderer = {
    backend: {
      device,
      createStorageAttribute(attribute: any) {
        if (!attributes.has(attribute)) {
          attributes.set(attribute, {
            buffer: device.createBuffer({ size: attribute.array.byteLength, usage: BUFFER_USAGE.STORAGE }),
          });
        }
      },
      get: (attribute: any) => attributes.get(attribute),
      destroyAttribute: (attribute: any) => attributes.delete(attribute),
    },
  };
  return { renderer, createComputePipelineAsync };
}

describe('reference pipeline reuse', () => {
  it('reuses compiled programs after resizing away and back, with fresh graph buffers', async () => {
    const { renderer, createComputePipelineAsync } = recordingRenderer();
    const model = await loadReferenceModel(renderer, source);
    const build = (width: number, height: number) => ReferenceWgslBackend.create({ renderer, model, width, height });
    const first = await build(64, 64);
    const firstFeatures = first.features;
    expect(createComputePipelineAsync.mock.calls.length).toBeGreaterThan(0);
    first.dispose();

    createComputePipelineAsync.mockClear();
    const resized = await build(128, 96);
    expect(createComputePipelineAsync.mock.calls.length).toBeGreaterThan(0);
    expect(resized.geometry.validWidth).toBe(128);
    resized.dispose();

    createComputePipelineAsync.mockClear();
    const restored = await build(64, 64);
    expect(restored.dispatchCount).toBe(451);
    expect(restored.features).not.toBe(firstFeatures);
    expect(createComputePipelineAsync.mock.calls.length).toBe(0);
    restored.dispose();
    model.dispose();
  });

  it('compiles independently on a new device, as after a page reload', async () => {
    for (let i = 0; i < 2; i++) {
      const { renderer, createComputePipelineAsync } = recordingRenderer();
      const backend = await ReferenceWgslBackend.create({ renderer, model: source, width: 64, height: 64 });
      expect(createComputePipelineAsync.mock.calls.length).toBeGreaterThan(0);
      backend.dispose();
    }
  });

  it('does not reuse different extra shader sources with the same name and entry point', async () => {
    const { renderer, createComputePipelineAsync } = recordingRenderer();
    const model = await loadReferenceModel(renderer, source);
    for (const body of ['let value = 1u;', 'let value = 2u;']) {
      createComputePipelineAsync.mockClear();
      const backend = await ReferenceWgslBackend.create({
        renderer,
        model,
        width: 64,
        height: 64,
        extraShaders: [
          { name: 'extra', code: `@compute @workgroup_size(1) fn extra() { ${body} }`, entryPoints: ['extra'] },
        ],
      });
      expect(
        createComputePipelineAsync.mock.calls.some(([descriptor]) => descriptor.compute.entryPoint === 'extra'),
      ).toBe(true);
      backend.dispose();
    }
    model.dispose();
  });

  it('retries shader preparation after a compilation failure', async () => {
    const { renderer, createComputePipelineAsync } = recordingRenderer();
    createComputePipelineAsync.mockRejectedValueOnce(new Error('Shader compilation failed'));
    const options = { renderer, model: source, width: 64, height: 64 };
    await expect(ReferenceWgslBackend.create(options)).rejects.toThrow('Shader compilation failed');
    const backend = await ReferenceWgslBackend.create(options);
    expect(backend.dispatchCount).toBe(451);
    backend.dispose();
  });
});
