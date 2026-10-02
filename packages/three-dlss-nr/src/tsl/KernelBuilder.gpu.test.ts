// KernelBuilder on the GPU: generated WGSL is shared between kernels that differ only in their buffers, outputs that
// a kernel does not write keep the sentinel (R6), and a rejected pass throws instead of failing silently.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { localId, workgroupId } from 'three/tsl';

import { fillSentinel, SENTINEL_BYTE } from '../../test/compare.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { expectIntegerComparisons, normalizeWGSL } from '../../test/wgsl.js';
import { createTensor, readBuffer, writeBuffer } from '../tensors.js';
import type { NRTensor } from '../types.js';
import { kernel, kernelWGSL, runKernels } from './KernelBuilder.js';
import { loadE4, u } from './packed.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

/** Copy the first `groups * 64` words of `input` to `output`, adding the low byte of each word. */
const copyKernel = (input: NRTensor, output: NRTensor, groups: number) =>
  kernel({
    label: `copy ${input.label}`,
    kind: 'test_copy',
    workgroupSize: [64],
    dispatch: [groups],
    inputs: { source: input },
    outputs: { target: output },
    body: ({ source, target }) => {
      const index = workgroupId.x.mul(u(64)).add(localId.x);
      target.element(index).assign(source.element(index).add(loadE4(source, index.mul(u(4)))));
    },
  });

describe('KernelBuilder', () => {
  it('kernels that differ only in buffers generate identical WGSL and share one program', async () => {
    const tensors = [0, 1, 2, 3].map((n) => createTensor(`t${n}`, 64, 4, 'e4'));
    const a = copyKernel(tensors[0], tensors[1], 1);
    const b = copyKernel(tensors[2], tensors[3], 1);
    const wgslA = kernelWGSL(gpu.renderer, a);
    expect(wgslA).toBe(kernelWGSL(gpu.renderer, b));
    expect(wgslA).toContain('var<storage, read> nr_source');
    expect(wgslA).toContain('var<storage, read_write> nr_target');
    expectIntegerComparisons(wgslA);
    expect(normalizeWGSL(wgslA)).not.toMatch(/NodeBuffer_\d/);
    await runKernels(gpu.renderer, [a, b]);
    const programs = gpu.renderer._pipelines.programs.compute;
    const shared = [...programs.keys()].filter((code: string) => code === wgslA);
    expect(shared).toHaveLength(1);
  });

  it('runs kernels in order in one pass; unwritten words keep the sentinel', async () => {
    // 128 rows x 4 E4 channels = 128 words.
    const input = createTensor('in', 128, 4, 'e4');
    const middle = createTensor('middle', 128, 4, 'e4');
    const output = createTensor('out', 128, 4, 'e4');
    writeBuffer(
      input,
      Uint32Array.from({ length: 128 }, (_, i) => i * 0x01010101),
    );
    fillSentinel(middle);
    fillSentinel(output);
    // The second kernel reads what the first wrote in the same pass, and writes only the first 64 words.
    await runKernels(gpu.renderer, [copyKernel(input, middle, 2), copyKernel(middle, output, 1)]);
    const words = new Uint32Array((await readBuffer(gpu.renderer, output, { validOnly: false })).buffer);
    const sentinel = SENTINEL_BYTE * 0x01010101;
    for (let i = 0; i < 128; ++i) {
      const once = (i * 0x01010101 + (i & 0xff)) >>> 0;
      const twice = (once + (once & 0xff)) >>> 0;
      expect(words[i], `word ${i}`).toBe(i < 64 ? twice : sentinel);
    }
  });

  it('a pass the device rejects throws (validation error scope)', async () => {
    const big = createTensor('big', 64, 4, 'e4');
    const k = copyKernel(big, createTensor('big out', 64, 4, 'e4'), 1);
    // Simulate a rejected submission: a workgroup size above the device limit.
    const bad = kernel({
      label: 'too wide',
      kind: 'test',
      workgroupSize: [gpu.device.limits.maxComputeInvocationsPerWorkgroup * 2],
      dispatch: [1],
      inputs: {},
      outputs: { target: big },
      body: ({ target }) => {
        target.element(localId.x).assign(u(1));
      },
    });
    await expect(runKernels(gpu.renderer, [k, bad])).rejects.toThrow(/rejected/);
  });
});
