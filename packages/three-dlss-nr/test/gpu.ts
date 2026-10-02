// GPU test harness: one Dawn device per test file, shared by our WebGPURenderer and the reference port.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Only for `*.gpu.test.ts` files, which run
// in the `gpu` vitest project (vitest-environment-webgpu-node: navigator.gpu backed by Dawn, no browser).

import { createCanvas } from 'vitest-environment-webgpu-node';
import { If, workgroupId, localId } from 'three/tsl';

import { createNRDevice, createNRRenderer, type NRDevice } from '../src/device.js';
import { kernel, runKernels, type BufferSource } from '../src/tsl/KernelBuilder.js';
import { u, type TSLNode } from '../src/tsl/packed.js';
import { attributeFromBytes, readBuffer, wordAttribute } from '../src/tensors.js';

export interface GpuTestContext extends NRDevice {
  renderer: any;
}

/**
 * A device (adapter limits raised, `shader-f16` when available) and an initialized WebGPURenderer on it. Call once
 * per file (`beforeAll`) and `dispose` it in `afterAll`.
 */
export async function createGpuTestContext(): Promise<GpuTestContext & { dispose(): void }> {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable: run the gpu project (vitest --project gpu)');
  const nr = await createNRDevice();
  const { renderer } = await createNRRenderer({ device: nr.device, canvas: createCanvas(1, 1).asElement() });
  return {
    ...nr,
    renderer,
    dispose() {
      renderer.dispose();
      nr.device.destroy();
    },
  };
}

/** A read-only input buffer holding `words`. */
export const wordsBuffer = (words: Uint32Array): BufferSource => ({ attribute: attributeFromBytes(words) });

/**
 * Run `count` independent invocations (`index` = 0..count-1, workgroups of 64 over x then y) and return the u32 each
 * one stored with `compute(index, inputs)`. `inputs` are bound read-only under their names.
 */
export async function runPerElement(
  renderer: any,
  {
    label,
    count,
    inputs = {},
    compute,
  }: {
    label: string;
    count: number;
    inputs?: Record<string, BufferSource>;
    compute: (index: TSLNode, views: Record<string, TSLNode>) => TSLNode;
  },
): Promise<Uint32Array> {
  const groups = Math.ceil(count / 64);
  const dispatch = groups <= 65535 ? [groups, 1, 1] : [65535, Math.ceil(groups / 65535), 1];
  const results = { attribute: wordAttribute(count * 4) };
  const k = kernel({
    label,
    kind: 'test',
    workgroupSize: [64],
    dispatch,
    inputs,
    outputs: { results },
    body: (views) => {
      const group = workgroupId.x.add(workgroupId.y.mul(u(65535)));
      const index = group.mul(u(64)).add(localId.x).toVar();
      If(index.lessThan(u(count)), () => {
        views.results.element(index).assign(compute(index, views as Record<string, TSLNode>));
      });
    },
  });
  await runKernels(renderer, [k]);
  const bytes = await readBuffer(renderer, results, { validOnly: false });
  return new Uint32Array(bytes.buffer, bytes.byteOffset, count);
}
