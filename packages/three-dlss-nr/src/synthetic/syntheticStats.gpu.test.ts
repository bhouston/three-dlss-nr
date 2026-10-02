// Calibration gate of the synthetic weights: the reference network (OpenDLSS-NR's WebGPU port, by maan, MIT) on the
// synthetic model keeps every block boundary in a useful range - not saturated, not collapsed to zero, many distinct
// codes - so that byte parity on these weights actually exercises the arithmetic. `synthetic/gains.ts` changes only
// together with this test.
//
// Needs the reference network, hence `shader-f16`; skipped where the device lacks it. The full network at 512x512
// on a software rasterizer takes many minutes, so on CI it also needs NR_FULL=1.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { referenceSupportsF16 } from '../../test/reference/refKernels.js';
import { createReferenceNetwork, loadReferenceModel } from '../../test/reference/refNetwork.js';
import { registerSyntheticFiles, unregisterSyntheticFiles } from '../../test/setup/fetchShim.js';
import { geometryFromValid } from '../geometry.js';
import { e4m3ToNumber } from '../numerics/oracle.js';
import { syntheticFeatures } from './features.js';
import { generateSyntheticModel } from './generate.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => {
  unregisterSyntheticFiles('nr-stats');
  gpu?.dispose();
});

function unavailable(): string | undefined {
  if (!referenceSupportsF16(gpu.device)) return 'the reference network needs shader-f16, which this device lacks';
  if (process.env.CI && !process.env.NR_FULL) return 'the full reference network on CI runs only with NR_FULL=1';
  return undefined;
}

/** Statistics of one E4 boundary tensor. */
function e4Stats(bytes: Uint8Array) {
  let saturated = 0;
  let zeros = 0;
  const distinct = new Set<number>();
  const magnitudes = new Float64Array(bytes.length);
  bytes.forEach((code, i) => {
    if ((code & 0x7f) === 0x7e) saturated += 1;
    if ((code & 0x7f) === 0) zeros += 1;
    distinct.add(code);
    magnitudes[i] = Math.abs(e4m3ToNumber(code));
  });
  magnitudes.sort();
  return {
    saturated: saturated / bytes.length,
    zeros: zeros / bytes.length,
    median: magnitudes[magnitudes.length >> 1],
    distinct: distinct.size,
  };
}

describe('synthetic weights calibration', () => {
  it.for([[512, 512]] as const)(
    'keeps every boundary of the reference network in range at %ix%i',
    async ([width, height], context) => {
      const reason = unavailable();
      if (reason) return context.skip(reason);
      const synthetic = await generateSyntheticModel({ seed: 1 });
      const model = await loadReferenceModel(gpu.device, registerSyntheticFiles('nr-stats', synthetic.files));
      const network = await createReferenceNetwork({
        device: gpu.device,
        model,
        width,
        height,
        captureBoundaries: true,
      });
      try {
        network.writeFeatures(syntheticFeatures(geometryFromValid(width, height)));
        await network.run();
        const failures: string[] = [];
        for (const name of network.boundaryNames) {
          const stats = e4Stats(await network.readBoundary(name));
          const line = `${name}: saturated ${(stats.saturated * 100).toFixed(3)}%, zeros ${(stats.zeros * 100).toFixed(1)}%, median ${stats.median}, ${stats.distinct} codes`;
          console.info(`[calibration] ${line}`);
          if (
            stats.saturated >= 0.001 ||
            stats.zeros >= 0.3 ||
            stats.median < 2 ** -4 ||
            stats.median > 8 ||
            stats.distinct < 40
          ) {
            failures.push(line);
          }
        }
        expect(failures).toEqual([]);
        const head = await network.readHead();
        expect(head.every(Number.isFinite)).toBe(true);
      } finally {
        network.destroy();
        model.destroy();
      }
    },
  );
});
