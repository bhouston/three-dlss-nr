// The whole TSL network end to end on the synthetic model, in Node (Dawn), without the reference: no validation
// errors, a finite head, repeated runs identical, and every block boundary, post tensor and the head equal to golden
// hashes recorded where the reference *could* run - the Chrome parity gate
// (test/browser/run-network-parity-chrome.mjs: the reference WebGPU port of OpenDLSS-NR, by maan, MIT, and this port on
// one device in real Chrome, byte for byte). CI's lavapipe has no trustworthy f16 for the reference, but our TSL never
// uses f16, so these hashes gate regressions everywhere.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { backendProblems, factoryProblems } from '../test/backendConformance.js';
import { createGpuTestContext, type GpuTestContext } from '../test/gpu.js';
import { NETWORK_GOLDEN, networkDigest } from '../test/network/golden.js';
import { NRModel } from './model/Model.js';
import { NRNetwork, tslBackend } from './NRNetwork.js';
import { syntheticFeatures } from './synthetic/features.js';
import { generateSyntheticModel } from './synthetic/generate.js';

let gpu: GpuTestContext & { dispose(): void };
let model: NRModel;

beforeAll(async () => {
  gpu = await createGpuTestContext();
  model = await NRModel.load(await generateSyntheticModel({ seed: 1 }));
}, 120_000);

afterAll(() => {
  model?.dispose();
  gpu?.dispose();
});

const sizes =
  process.env.CI && !process.env.NR_FULL
    ? ([[64, 64]] as const)
    : ([
        [64, 64],
        [512, 512],
      ] as const);

describe('NRNetwork on synthetic weights', () => {
  it.for(sizes)(
    'runs the whole network at %ix%i, deterministically, equal to the Chrome-verified hashes',
    { timeout: 1_800_000 },
    async ([width, height]) => {
      const started = performance.now();
      const network = await NRNetwork.create({
        renderer: gpu.renderer,
        model,
        width,
        height,
        captureBoundaries: true,
      });
      try {
        const compiled = performance.now();
        expect(backendProblems(network)).toEqual([]);
        expect(factoryProblems(tslBackend)).toEqual([]);
        expect(tslBackend.unavailableReason(gpu.renderer)).toBeNull();
        expect(network.dispatchLabels).toHaveLength(451);
        expect(network.dispatchCount).toBe(451 + 79);
        network.writeFeatures(syntheticFeatures(network.geometry));
        await network.run();
        const first = await networkDigest(network);
        const head = await network.readHead();
        expect(head.every(Number.isFinite)).toBe(true);
        await network.run();
        const second = await networkDigest(network);
        expect(second).toEqual(first);
        console.info(
          `[network ${width}x${height}] create ${((compiled - started) / 1000).toFixed(1)} s ` +
            '(timings unreliable: GPU shared); digest ' +
            first.all,
        );
        const golden = NETWORK_GOLDEN[`${width}x${height}`];
        expect(golden, `no golden digest for ${width}x${height}`).toBeDefined();
        expect(first.tensors).toEqual(golden.tensors);
      } finally {
        network.dispose();
      }
    },
  );
});
