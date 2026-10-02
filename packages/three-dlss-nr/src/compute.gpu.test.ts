import { expect, it } from 'vitest';
import { Fn, instanceIndex, instancedArray, uint } from 'three/tsl';
import { WebGPURenderer } from 'three/webgpu';
import { createCanvas } from 'vitest-environment-webgpu-node';

// Smoke test for the `gpu` vitest project: runs a tiny TSL compute kernel on
// a real WebGPU device (Dawn, headless) and reads the result back.
it('runs a TSL compute kernel on WebGPU and reads the result back', async () => {
  expect(navigator.gpu, 'WebGPU is unavailable').toBeDefined();
  const renderer = new WebGPURenderer({ canvas: createCanvas(1, 1).asElement(), antialias: false });
  await renderer.init();
  try {
    // A real WebGPU device, not the WebGL2 fallback backend.
    expect(renderer.backend.device, 'WebGPURenderer fell back to WebGL').toBeDefined();
    const count = 64;
    const buffer = instancedArray(count, 'uint');
    renderer.compute(
      Fn(() => {
        buffer.element(instanceIndex).assign(instanceIndex.mul(instanceIndex).add(uint(7)));
      })().compute(count),
    );
    const out = new Uint32Array(await renderer.getArrayBufferAsync(buffer.value));
    expect(Array.from(out)).toEqual(Array.from({ length: count }, (_, i) => i * i + 7));
  } finally {
    renderer.dispose();
  }
});
