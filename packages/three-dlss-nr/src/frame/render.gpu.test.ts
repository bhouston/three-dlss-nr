// The frame kernels fed straight from a WebGPURenderer render (design 4.4): an MRT render target with the colour and
// three's velocity, read by `createInputFeatures` as textures. Checks the conventions the texture path relies on:
// texture row 0 is the top of the image, and three's velocity is current NDC minus previous NDC with y up.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Color, HalfFloatType, Mesh, OrthographicCamera, PlaneGeometry, Scene } from 'three';
import { MeshBasicNodeMaterial, RenderTarget } from 'three/webgpu';
import { mrt, output, velocity } from 'three/tsl';

import { f16ToNumber } from '../numerics/oracle.js';
import { createTensor, readBuffer } from '../tensors.js';
import { runKernels } from '../tsl/KernelBuilder.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { NRFrameParams } from './frameInputs.js';
import { NRHistory } from './history.js';
import { createInputFeatures } from './inputFeatures.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

describe('input features from an MRT render (colour + velocity textures)', () => {
  it('reads the render top-down and three velocity as current - previous NDC, y up', async () => {
    const [width, height] = [32, 24];
    const renderer = gpu.renderer;
    const target = new RenderTarget(width, height, { count: 2, type: HalfFloatType, depthBuffer: true });
    target.textures[0].name = 'output';
    target.textures[1].name = 'velocity';

    // An orthographic view of [-1, 1]^2; a red quad over the top half.
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 5;
    const scene = new Scene();
    scene.background = new Color(0, 0, 0);
    const quad = new Mesh(new PlaneGeometry(1, 1), new MeshBasicNodeMaterial({ color: 0xff0000 }));
    quad.position.set(0, 0.5, 0);
    scene.add(quad);

    renderer.setMRT(mrt({ output, velocity }));
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    // Move the quad right by 0.25 NDC between two frames: velocity.x = +0.25 on it.
    quad.position.x += 0.25;
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.setMRT(null);

    const velocityBits = new Uint16Array(
      (await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height, 1)).buffer,
    );
    // The quad's centre is now at NDC (0.25, 0.5): pixel (20, 6) from the top.
    const at = (x: number, y: number, c: number) => f16ToNumber(velocityBits[(y * width + x) * 4 + c]);
    expect(at(20, 6, 0)).toBeCloseTo(0.25, 3);
    expect(at(20, 6, 1)).toBeCloseTo(0, 3);

    const features = createTensor('input features', width * height, 16, 'f32');
    const params = new NRFrameParams({ historyValid: false });
    const k = createInputFeatures(
      { fullWidth: width, fullHeight: height, validWidth: width, validHeight: height, rejectOffscreenHistory: true },
      {
        color: { texture: target.textures[0] },
        motion: { velocityTexture: target.textures[1] },
        history: new NRHistory(width, height).read(0),
        features,
        params,
      },
    );
    await runKernels(renderer, [k]);
    const lanes = new Float32Array((await readBuffer(renderer, features)).buffer);
    const red = (x: number, y: number) => lanes[(y * width + x) * 16 + 4];
    const green = (x: number, y: number) => lanes[(y * width + x) * 16 + 5];
    // Row 6 (top half) on the quad is red; row 18 (bottom half) is the black background (centred code -0.0625).
    expect(red(20, 6)).toBeGreaterThan(0.05);
    expect(green(20, 6)).toBe(-0.0625);
    expect(red(20, 18)).toBe(-0.0625);
  });
});
