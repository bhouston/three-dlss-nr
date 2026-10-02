// The reference `Model` (OpenDLSS-NR's ports/browser-webgpu/src/model.js, by maan, MIT) loads the synthetic model
// on a real Dawn device through the fetch shim (`Model.load('synthetic://nr')`), and every buffer it builds for the
// graph, read back from the GPU, equals the NRModel attribute for the same request byte for byte. The reference
// Model has no f16 shaders, so this runs on every device.

import { readBack } from '@ref/gpu.js';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import {
  compareRequests,
  loadReferenceModelOn,
  readableDevice,
  recordGraphRequests,
} from '../../test/reference/refModel.js';
import { registerSyntheticFiles, unregisterSyntheticFiles } from '../../test/setup/fetchShim.js';
import { generateSyntheticModel } from '../synthetic/generate.js';
import { NRModel } from './Model.js';

let gpu: GpuTestContext & { dispose(): void };
let reference: any;
let ours: NRModel;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  const synthetic = await generateSyntheticModel({ seed: 1 });
  ours = await NRModel.load(synthetic, { verify: true });
  const directory = registerSyntheticFiles('nr', synthetic.files);
  expect(directory).toBe('synthetic://nr');
  reference = await loadReferenceModelOn(readableDevice(gpu.device), directory);
});

afterAll(() => {
  reference?.destroy();
  unregisterSyntheticFiles('nr');
  gpu?.dispose();
});

describe('reference Model on the GPU', () => {
  it("loads 'synthetic://nr' and uploads every stage unchanged", async () => {
    expect(reference.blockCount).toBe(71);
    expect(reference.tensors.size).toBe(153);
    expect(reference.blendScale()).toBe(0.75);
    for (const [id, bytes] of ours.stages) {
      const buffer = reference.stages.get(id);
      const uploaded = new Uint8Array(await readBack(gpu.device, buffer, buffer.size));
      expect(uploaded.length).toBe((bytes.length + 4 + 3) & ~3);
      expect(Buffer.from(uploaded.subarray(0, bytes.length)).equals(Buffer.from(bytes)), id).toBe(true);
    }
  });

  it('builds, for every weight request of the graph, the buffers NRModel builds', async () => {
    const { requests, dispatches } = recordGraphRequests(reference, 64, 64);
    expect(dispatches).toHaveLength(451);
    const { compared, mismatches } = await compareRequests(
      requests,
      reference,
      ours,
      async (buffer: GPUBuffer) => new Uint8Array(await readBack(gpu.device, buffer, buffer.size)),
    );
    expect(mismatches).toEqual([]);
    expect(compared).toBe(638);
  });
});
