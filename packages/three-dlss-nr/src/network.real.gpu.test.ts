// The TSL network on real weights against recorded fixtures, with the reference's own fixture contract.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). NVIDIA's weights are proprietary and are not in
// this repository: this test runs only where someone has them, behind two environment variables:
//   NR_WEIGHTS   a model directory (manifest.json + model/stages/*, the reference's docs/weights.md layout);
//   NR_FIXTURES  a fixture directory (one with manifest.json, e.g. nr512) or a directory of them (nr512, nr768).
// Each fixture is loaded and checked by the reference's ports/browser-webgpu/src/parity.js `loadFixture` /
// `runParity` (which reject a malformed or truncated fixture before running), driving `NRNetwork` through a thin
// adapter that gives it the reference `Network`'s shape. Features come from the fixture, or from its proxy through
// the TSL port of preprocess.wgsl (`createPreprocess`), as `Network.featuresFromProxy` does.

import { existsSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { loadFixture, runParity } from '@ref/parity.js';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createGpuTestContext, type GpuTestContext } from '../test/gpu.js';
import { createPreprocess } from './kernels/preprocess.js';
import { NRModel } from './model/Model.js';
import { NRNetwork } from './NRNetwork.js';
import { attributeFromBytes } from './tensors.js';
import { runKernels } from './tsl/KernelBuilder.js';

const weights = process.env.NR_WEIGHTS;
const fixturesRoot = process.env.NR_FIXTURES;

/** Fixture directories: NR_FIXTURES itself when it holds a manifest, else its subdirectories that do. */
function fixtureDirectories(): string[] {
  if (!fixturesRoot) return [];
  const root = resolve(fixturesRoot);
  if (existsSync(join(root, 'manifest.json'))) return [root];
  return readdirSync(root)
    .map((name) => join(root, name))
    .filter((dir) => existsSync(join(dir, 'manifest.json')))
    .toSorted();
}

const fixtures = fixtureDirectories();
let gpu: GpuTestContext & { dispose(): void };
let model: NRModel;

beforeAll(async () => {
  if (!weights || !fixtures.length) return;
  gpu = await createGpuTestContext();
  model = await NRModel.load(pathToFileURL(resolve(weights)).href);
}, 600_000);

afterAll(() => {
  model?.dispose();
  gpu?.dispose();
});

/** The reference `Network` surface `runParity` uses, over an `NRNetwork`. */
function referenceShaped(network: NRNetwork) {
  return {
    geometry: network.geometry,
    writeFeatures: (data: Float32Array) => network.writeFeatures(data),
    async featuresFromProxy(proxy: Float32Array, manifest: any) {
      const g = network.geometry;
      const conditioning = manifest.conditioning ?? {};
      const kernel = createPreprocess(
        {
          fullWidth: g.fullWidth,
          fullHeight: g.fullHeight,
          validWidth: g.validWidth,
          validHeight: g.validHeight,
          sourceWidth: manifest.proxy.width,
          sourceHeight: manifest.proxy.height,
          seed: manifest.seed ?? 0,
          autoMask: !!manifest.autoMask,
          localTone: conditioning.localTone ?? 1,
          localStructure: conditioning.localStructure ?? 1,
          skinStructure: conditioning.skinStructure ?? -1,
          style: conditioning.style ?? 0,
        },
        { proxy: { attribute: attributeFromBytes(proxy) }, features: network.features },
      );
      await runKernels(network.renderer, [kernel]);
    },
    run: () => network.run(),
    recorder: { dispatchCount: network.dispatchCount, enableProfiling: () => {}, readProfile: async () => null },
    boundaryNames: network.boundaryNames,
    readBoundary: (name: string) => network.readBoundary(name),
    readHead: () => network.readHead(),
  };
}

describe('NRNetwork on real weights (NR_WEIGHTS, NR_FIXTURES)', () => {
  it.skipIf(!weights || !fixtures.length).for(fixtures.length ? fixtures : ['(none)'])(
    'matches fixture %s',
    { timeout: 3_600_000 },
    async (directory) => {
      const fixture = await loadFixture(pathToFileURL(directory).href.replace(/\/+$/, ''));
      const [width, height] = fixture.manifest.sourceDimensions;
      const network = await NRNetwork.create({ renderer: gpu.renderer, model, width, height, captureBoundaries: true });
      const rows: string[] = [];
      const failures: string[] = [];
      const report = {
        note: (text: string) => rows.push(text),
        status: () => {},
        timing: (ms: number) => rows.push(`frame ${ms.toFixed(0)} ms (unreliable: GPU shared)`),
        row: (name: string, result: any, shape: string) => {
          const verdict = typeof result === 'string' ? result : result.verdict;
          rows.push(`${name} ${shape}: ${verdict}`);
          if (verdict !== 'bit-exact' && verdict !== 'within-tolerance') failures.push(`${name}: ${verdict}`);
        },
        done: () => {},
      };
      try {
        const summary = await runParity(referenceShaped(network), fixture, report);
        console.info(`[fixture ${directory}]\n  ${rows.join('\n  ')}`);
        expect(failures).toEqual([]);
        expect(summary.failed).toBe(0);
      } finally {
        network.dispose();
      }
    },
  );
});
