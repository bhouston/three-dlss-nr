// Synthetic numerical parity across reference graph rebuilds; no NVIDIA weights.
import { afterAll, beforeAll, expect, it } from 'vitest';
import { createGpuTestContext, type GpuTestContext } from '../test/gpu.js';
import { loadReferenceModel, ReferenceWgslBackend } from './reference-backend/ReferenceWgslBackend.js';
import { syntheticFeatures } from './synthetic/features.js';
import { generateSyntheticModel } from './synthetic/generate.js';

let gpu: GpuTestContext & { dispose(): void };
beforeAll(async () => {
  gpu = await createGpuTestContext();
});
afterAll(() => gpu?.dispose());

it('preserves synthetic output bytes when returning to a cached resolution', async (context) => {
  if (!gpu.shaderF16) return context.skip('requires shader-f16');
  const model = await loadReferenceModel(gpu.renderer, await generateSyntheticModel());
  let backend: ReferenceWgslBackend | undefined;
  try {
    const create = (width: number) => ReferenceWgslBackend.create({ renderer: gpu.renderer, model, width, height: 64 });
    backend = await create(64);
    const features = syntheticFeatures(backend.geometry);
    backend.writeFeatures(features);
    await backend.run();
    const cold = await backend.readHead();
    expect(cold.every(Number.isFinite)).toBe(true);
    backend.dispose();
    backend = await create(128);
    backend.dispose();
    backend = await create(64);
    backend.writeFeatures(features);
    await backend.run();
    const warm = await backend.readHead();
    expect(new Uint8Array(warm.buffer)).toEqual(new Uint8Array(cold.buffer));
  } finally {
    backend?.dispose();
    model.dispose();
  }
});
