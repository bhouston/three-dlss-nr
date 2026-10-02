// The reference-wgsl backend (OpenDLSS-NR's WebGPU port, by maan, MIT, run on three's device) under Dawn in Node.
//
// Where the device lacks `shader-f16` (Dawn in Node on the Windows dev machine) the backend must refuse with a clear
// reason. Where it has it, and a model directory is given (`NR_MODEL_DIR`, e.g. from
// `node scripts/make-synthetic-model.mjs <dir>`), the backend is created at 64x64 and checked for conformance,
// finite output and repeatable runs. Byte equality with the standalone reference is checked in real Chrome:
// test/browser/run-shim-parity-chrome.mjs (Dawn in Node cannot run the reference here, and lavapipe's half arithmetic
// is untrusted, see README-internals).

import { pathToFileURL } from 'node:url';

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { backendProblems } from '../../test/backendConformance.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { createTensor } from '../tensors.js';
import { referenceWgslBackend, sharedTensorBuffer } from './index.js';

let gpu: GpuTestContext & { dispose(): void };
beforeAll(async () => {
  gpu = await createGpuTestContext();
});
afterAll(() => gpu?.dispose());

describe('reference-wgsl backend on three.js device', () => {
  it('shares a tensor buffer with three (one GPUBuffer for both)', () => {
    const tensor = createTensor('shared', 64, 4, 'f32');
    const buffer = sharedTensorBuffer(gpu.renderer, tensor);
    expect(buffer.size).toBe(tensor.byteLength);
    expect(buffer.usage & 0x80).toBeTruthy(); // STORAGE
    // three adopts the existing buffer rather than creating another.
    gpu.renderer.backend.createStorageAttribute(tensor.attribute);
    expect(gpu.renderer.backend.get(tensor.attribute).buffer).toBe(buffer);
    gpu.renderer.backend.destroyAttribute(tensor.attribute);
  });

  it('refuses a device without shader-f16, naming the fix', async (context) => {
    if (gpu.shaderF16) return context.skip('this device has shader-f16');
    const reason = referenceWgslBackend.unavailableReason(gpu.renderer);
    expect(reason).toMatch(/'shader-f16'/);
    await expect(
      referenceWgslBackend.create({ renderer: gpu.renderer, model: 'unused', width: 64, height: 64 }),
    ).rejects.toThrow(/shader-f16/);
  });

  it('runs at 64x64 when the device has shader-f16 and NR_MODEL_DIR is set', async (context) => {
    if (!gpu.shaderF16) return context.skip('needs shader-f16');
    const dir = process.env.NR_MODEL_DIR;
    if (!dir) return context.skip('set NR_MODEL_DIR to a model directory');
    const backend = await referenceWgslBackend.create({
      renderer: gpu.renderer,
      model: pathToFileURL(dir).href,
      width: 64,
      height: 64,
    });
    try {
      expect(backendProblems(backend)).toEqual([]);
      expect(backend.dispatchCount).toBe(451);
      const features = new Float32Array(backend.geometry.fullRows * 16);
      for (let i = 0; i < features.length; ++i) features[i] = ((i * 2654435761) % 1024) / 1024 - 0.5;
      backend.writeFeatures(features);
      await backend.run();
      const first = await backend.readHead();
      await backend.run();
      expect(await backend.readHead()).toEqual(first);
      expect(first.every(Number.isFinite)).toBe(true);
    } finally {
      backend.dispose();
    }
  });
});
