// NRModel against the reference `Model` (OpenDLSS-NR's ports/browser-webgpu/src/model.js, by maan, MIT), without a
// GPU: the reference loads the synthetic model through the test fetch shim onto a recording fake device, a
// record-only pass of the reference graph makes every weight request, and each of our buffers must equal the
// reference's byte for byte.

import { beforeAll, describe, expect, it } from 'vitest';

import { registerSyntheticFiles } from '../../test/setup/fetchShim.js';
import {
  compareRequests,
  fakeDevice,
  fakeReader,
  loadReferenceModelOn,
  recordGraphRequests,
} from '../../test/reference/refModel.js';
import { generateSyntheticModel, type SyntheticModel } from '../synthetic/generate.js';
import { manifestProblems, parseManifest, type NRManifest } from './manifest.js';
import { NRModel } from './Model.js';

let synthetic: SyntheticModel;
let ours: NRModel;

beforeAll(async () => {
  synthetic = await generateSyntheticModel({ seed: 1 });
  ours = await NRModel.load(synthetic, { verify: true });
});

/** The synthetic model with one manifest or stage change. */
function variant(change: (manifest: any, files: Map<string, Uint8Array>) => void) {
  const manifest = structuredClone(synthetic.manifest) as any;
  const files = new Map(synthetic.files);
  change(manifest, files);
  files.set('manifest.json', new TextEncoder().encode(JSON.stringify(manifest)));
  return { files };
}

describe('NRModel', () => {
  it('loads the synthetic model from memory and from a URL', async () => {
    expect(ours.blockCount).toBe(71);
    expect(ours.tensors.size).toBe(153);
    expect(ours.blendScale()).toBe(0.75);
    expect(ours.tensor(31, 3).byteLength).toBe(2);
    expect(() => ours.tensor(71)).toThrow('missing tensor block71.layer0.layer');

    const url = registerSyntheticFiles('nr-unit', synthetic.files);
    let progress = 0;
    const fetched = await NRModel.load(url, { verify: true, onProgress: (loaded) => (progress = loaded) });
    expect(progress).toBe(synthetic.manifest.stages.reduce((sum, stage) => sum + stage.packedByteLength, 0));
    expect(fetched.manifest).toEqual(synthetic.manifest);
    await expect(NRModel.load('synthetic://nowhere')).rejects.toThrow('cannot read the manifest');
  });

  it('builds the same bytes as the reference Model for every weight request of the graph', async () => {
    const url = registerSyntheticFiles('nr-unit-ref', synthetic.files);
    const { device } = fakeDevice();
    const reference = await loadReferenceModelOn(device, url);
    const { requests, dispatches } = recordGraphRequests(reference, 64, 64);
    // The whole frame (graph.js:354-580), with every reference load-time check passed on the synthetic weights.
    expect(dispatches).toHaveLength(451);
    expect(new Set(requests.map((request) => request.method))).toEqual(
      new Set(['fp8Matrix', 'f16Matrix', 'relativeBias', 'headScales', 'auxVector', 'auxPair', 'auxOffset']),
    );
    const methods: Record<string, number> = {};
    for (const request of requests) methods[request.method] = (methods[request.method] ?? 0) + 1;
    expect(methods).toMatchInlineSnapshot(`
      {
        "auxOffset": 140,
        "auxPair": 1,
        "auxVector": 5,
        "f16Matrix": 2,
        "fp8Matrix": 358,
        "headScales": 70,
        "relativeBias": 62,
      }
    `);
    const { compared, mismatches } = await compareRequests(requests, reference, ours, fakeReader);
    expect(mismatches).toEqual([]);
    expect(compared).toBe(requests.length);
    // Weights do not depend on the frame size: another geometry asks for the same buffers.
    const other = recordGraphRequests(reference, 512, 512);
    expect(other.dispatches).toHaveLength(451);
    expect(other.requests.map((r) => `${r.method} ${r.tensor} ${JSON.stringify(r.args)}`)).toEqual(
      requests.map((r) => `${r.method} ${r.tensor} ${JSON.stringify(r.args)}`),
    );
  });

  it('caches each buffer by key', () => {
    const tensor = ours.tensor(5);
    expect(ours.fp8Matrix(tensor, 0, 128, 128, { batchK: 64 })).toBe(
      ours.fp8Matrix(tensor, 0, 128, 128, { batchK: 64 }),
    );
    expect(ours.relativeBias(tensor, 100, 2)).toBe(ours.relativeBias(tensor, 100, 2));
  });

  it('rejects what the reference rejects: unbounded weights and bad shapes', async () => {
    const entry = synthetic.manifest.tensors.find((tensor) => tensor.name === 'block1.layer0.layer')!;
    const corrupt = variant((_manifest, files) => {
      const path = `model/stages/${entry.stage}.bin`;
      const stage = files.get(path)!.slice();
      stage[entry.stageOffset + 7] = 0x52; // |w| = 10
      files.set(path, stage);
    });
    const model = await NRModel.load(corrupt);
    expect(() => model.fp8Matrix(model.tensor(1), 0, 32, 128)).toThrow('weight 7 of block1.layer0.layer');
    const url = registerSyntheticFiles('nr-unit-corrupt', corrupt.files);
    const reference = await loadReferenceModelOn(fakeDevice().device, url);
    expect(() => recordGraphRequests(reference, 64, 64)).toThrow('weight 7 of block1.layer0.layer');
    // The NaN code is allowed, as in the reference.
    const nan = variant((_manifest, files) => {
      const path = `model/stages/${entry.stage}.bin`;
      const stage = files.get(path)!.slice();
      stage[entry.stageOffset + 7] = 0xff;
      files.set(path, stage);
    });
    expect(() => {
      const withNaN = new NRModel(synthetic.manifest, stagesById(nan.files));
      withNaN.fp8Matrix(withNaN.tensor(1), 0, 32, 128);
    }).not.toThrow();

    expect(() => ours.fp8Matrix(ours.tensor(1), 0, 48, 128)).toThrow('FP8 matrix shape');
    expect(() => ours.fp8Matrix(ours.tensor(1), 20000, 32, 128)).toThrow('FP8 matrix exceeds tensor');
    expect(() => ours.f16Matrix(ours.tensor(0), 0, 8, 32)).toThrow('f16 matrix K must be a multiple of 16');
    expect(() => ours.relativeBias(ours.tensor(1), 20000, 1)).toThrow('relative bias exceeds tensor');
    expect(() => ours.auxVector(ours.tensor(1), 1, 4)).toThrow('not half-aligned');
  });

  it('validates the manifest and the stages', async () => {
    expect(problems((m) => (m.totals.blockCount = 70))).toEqual(['the model has 70 blocks; this network has 71']);
    expect(problems((m) => m.tensors.splice(5, 1))).toEqual(['missing tensor block5.layer0.layer']);
    // Exact lengths where the reference graph checks them, minimums elsewhere.
    expect(problems((m) => (m.tensors[0].byteLength -= 16))).toEqual([
      'tensor block0.layer0.layer is 21680 bytes; the graph needs exactly 21696',
    ]);
    expect(problems((m) => (m.tensors[1].byteLength -= 16))).toEqual([]);
    expect(problems((m) => (m.tensors[1].byteLength -= 17))).toEqual([
      'tensor block1.layer0.layer is 20655 bytes; the graph needs at least 20656',
    ]);
    expect(problems((m) => (m.tensors[0].stage = 'nope'))).toEqual([
      'tensor block0.layer0.layer references unknown stage nope',
    ]);
    expect(() => parseManifest({ totals: {}, stages: [], tensors: [] })).toThrow('totals.blockCount');
    expect(() => parseManifest({ totals: { blockCount: 71 }, stages: [{ id: 1 }], tensors: [] })).toThrow('stage 0');

    await expect(NRModel.load(variant((m) => (m.stages[3].packedByteLength += 4)))).rejects.toThrow(
      /exceeds stage|stage size mismatch/,
    );
    const tampered = variant((m) => (m.stages[2].sha256 = '0'.repeat(64)));
    await expect(NRModel.load(tampered, { verify: true })).rejects.toThrow('stage s02 fails its sha256 check');
    await expect(NRModel.load(tampered)).resolves.toBeInstanceOf(NRModel);
  });
});

/** The problems of the synthetic manifest after one change. */
function problems(change: (manifest: any) => void): string[] {
  const manifest = structuredClone(synthetic.manifest) as any;
  change(manifest);
  return manifestProblems(parseManifest(manifest) as NRManifest);
}

function stagesById(files: ReadonlyMap<string, Uint8Array>): Map<string, Uint8Array> {
  return new Map(synthetic.manifest.stages.map((stage) => [stage.id, files.get(`model/${stage.file}`)!]));
}
