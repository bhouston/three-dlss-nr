// DlssNrPass end to end in Node (Dawn) on a small scene: the NR-off path, the frame kernels around a stub backend
// (no shader-f16 needed), split view, history resets, rebuilds, and an external frame.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Color, Mesh, OrthographicCamera, PlaneGeometry, Scene, UnsignedByteType } from 'three';
import { DataTexture, MeshBasicNodeMaterial, RenderTarget } from 'three/webgpu';
import { localId, workgroupId, If } from 'three/tsl';

import type { NRBackend, NRFrameTiming } from '../backend/NRBackend.js';
import { createTensor } from '../tensors.js';
import { kernel } from '../tsl/KernelBuilder.js';
import { u } from '../tsl/packed.js';
import { geometryFromValid } from '../geometry.js';
import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { clampSize, DlssNrPass, type DlssNrExternalFrame } from './DlssNrPass.js';

let gpu: GpuTestContext & { dispose(): void };

beforeAll(async () => {
  gpu = await createGpuTestContext();
});

afterAll(() => gpu?.dispose());

const [WIDTH, HEIGHT] = [64, 48];

/** An orthographic view of [-1, 1]^2: a grey background, a red quad over the top half. */
function testScene() {
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 5;
  const scene = new Scene();
  scene.background = new Color(0.2, 0.2, 0.2);
  const quad = new Mesh(new PlaneGeometry(2, 1), new MeshBasicNodeMaterial({ color: new Color(0.8, 0.1, 0.05) }));
  quad.position.set(0, 0.5, 0);
  scene.add(quad);
  return { scene, camera, quad };
}

/** A backend stand-in: writes a constant head (red residual +0.5, blend logit 0) and counts its runs. */
function stubBackend(renderer: any, width: number, height: number): NRBackend & { runs: number; disposed: boolean } {
  const geometry = geometryFromValid(width, height);
  const features = createTensor('input features', geometry.fullRows, 16, 'f32');
  const head = createTensor('head', geometry.fullRows, 4, 'f32');
  const red = new Float32Array([0.5]);
  const redBits = new Uint32Array(red.buffer)[0];
  const rows = geometry.fullRows;
  const fill = kernel({
    label: 'stub head',
    kind: 'stub_head',
    workgroupSize: [64],
    dispatch: [Math.ceil(rows / 64)],
    inputs: { features },
    outputs: { head },
    body: ({ features: f, head: h }) => {
      const row = workgroupId.x.mul(u(64)).add(localId.x).toVar();
      If(row.lessThan(u(rows)), () => {
        // Touch the features so they stay bound (and are read after input features wrote them).
        const keep = f.element(row.mul(u(16)).add(u(3))).bitAnd(u(0));
        h.element(row.mul(u(4))).assign(u(redBits).bitOr(keep));
        h.element(row.mul(u(4)).add(u(1))).assign(u(0));
        h.element(row.mul(u(4)).add(u(2))).assign(u(0));
        h.element(row.mul(u(4)).add(u(3))).assign(u(0));
      });
    },
  });
  const backend = {
    id: 'tsl' as const,
    label: 'stub',
    requirements: { features: [], limits: {} },
    geometry,
    dispatchCount: 1,
    features,
    head,
    boundaryNames: [],
    memory: { activationBytes: 0, weightBytes: 0 },
    blendScale: 0.75,
    runs: 0,
    disposed: false,
    writeFeatures() {},
    async run(): Promise<NRFrameTiming> {
      backend.runs += 1;
      renderer.compute(fill.node);
      await renderer.backend.device.queue.onSubmittedWorkDone();
      return { method: 'submitted-work-done', gpuMilliseconds: null, wallMilliseconds: 0 };
    },
    readHead: async () => new Float32Array(0),
    readBoundary: async () => new Uint8Array(0),
    readTensor: async () => new Uint8Array(0),
    dispose() {
      backend.disposed = true;
    },
  };
  return backend;
}

const row3 = (m: number[][], v: number[]) => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
const aces = (x: number) => (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.432951) + 0.238081);

/** The reference's display transform (frame.wgsl `display_transform`) on the CPU, for one linear colour. */
function displayTransform([r, g, b]: number[]): number[] {
  const input = row3(
    [
      [0.59719, 0.35458, 0.04823],
      [0.076, 0.90834, 0.01566],
      [0.0284, 0.13383, 0.83777],
    ],
    [r / 0.6, g / 0.6, b / 0.6],
  ).map(aces);
  const out = row3(
    [
      [1.60475, -0.53108, -0.07367],
      [-0.10208, 1.10813, -0.00605],
      [-0.00327, -0.07276, 1.07602],
    ],
    input,
  ).map((x) => Math.min(1, Math.max(0, x)));
  return out.map((x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055));
}

async function withValidation<T>(body: () => Promise<T>): Promise<T> {
  const device: GPUDevice = gpu.renderer.backend.device;
  device.pushErrorScope('validation');
  const result = await body();
  const error = await device.popErrorScope();
  expect(error?.message ?? null).toBeNull();
  return result;
}

/** Render one frame of `pass` into an RGBA8 target and read it back (top row first). */
async function frame(pass: DlssNrPass, present: any) {
  const renderer = gpu.renderer;
  renderer.setRenderTarget(present);
  const stats = await withValidation(() => pass.render());
  renderer.setRenderTarget(null);
  const pixels: Uint8Array = await renderer.readRenderTargetPixelsAsync(present, 0, 0, WIDTH, HEIGHT);
  return { stats, pixels: new Uint8Array(pixels.buffer, pixels.byteOffset, WIDTH * HEIGHT * 4) };
}

const pixel = (pixels: Uint8Array, x: number, y: number) =>
  Array.from(pixels.subarray((y * WIDTH + x) * 4, (y * WIDTH + x) * 4 + 3));

describe('DlssNrPass', () => {
  it('clamps sizes to 1280x720 pixels and at least 64', () => {
    expect(clampSize(640, 360)).toEqual({ width: 640, height: 360 });
    expect(clampSize(2560, 1440)).toEqual({ width: 1280, height: 720 });
    expect(clampSize(10, 10)).toEqual({ width: 64, height: 64 });
  });

  it('presents the display transform of the scene with NR off (no network)', async () => {
    const { scene, camera } = testScene();
    const pass = new DlssNrPass({ renderer: gpu.renderer, scene, camera, width: WIDTH, height: HEIGHT });
    const present = new RenderTarget(WIDTH, HEIGHT, { type: UnsignedByteType });
    expect(pass.state).toBe('none');
    const { stats, pixels } = await frame(pass, present);
    expect(stats?.ran).toBe(false);
    // Top half: the quad; bottom half: the background. Image row 0 is the top.
    const expectQuad = displayTransform([0.8, 0.1, 0.05]).map((x) => x * 255);
    const expectBackground = displayTransform([0.2, 0.2, 0.2]).map((x) => x * 255);
    for (const [x, y, expected] of [
      [32, 8, expectQuad],
      [32, 40, expectBackground],
    ] as const) {
      const got = pixel(pixels, x, y);
      for (let c = 0; c < 3; ++c) expect(Math.abs(got[c] - expected[c])).toBeLessThanOrEqual(2);
    }
    pass.dispose();
    present.dispose();
  });

  it('runs input features, a backend and compose; split view; history resets; rebuilds on resize', async () => {
    const { scene, camera, quad } = testScene();
    const pass = new DlssNrPass({ renderer: gpu.renderer, scene, camera, width: WIDTH, height: HEIGHT, view: 'off' });
    const present = new RenderTarget(WIDTH, HEIGHT, { type: UnsignedByteType });
    const built: ReturnType<typeof stubBackend>[] = [];
    const states: string[] = [];
    pass.onStatus = (state) => states.push(state);
    await pass.setNetwork(async ({ renderer, width, height }) => {
      const backend = stubBackend(renderer, width, height);
      built.push(backend);
      return backend;
    });
    expect(pass.state).toBe('ready');
    expect(states).toEqual(['building', 'ready']);

    const off = await frame(pass, present);
    expect(off.stats?.ran).toBe(false);
    expect(built[0].runs).toBe(0);

    pass.view = 'on';
    const first = await frame(pass, present);
    expect(first.stats?.ran).toBe(true);
    expect(first.stats?.historyReset).toBe(true);
    expect(built[0].runs).toBe(1);
    // The red residual makes the quad and the background redder than with NR off.
    for (const [x, y] of [
      [32, 8],
      [32, 40],
    ]) {
      expect(pixel(first.pixels, x, y)[0]).toBeGreaterThan(pixel(off.pixels, x, y)[0] + 10);
    }
    quad.position.x += 0.1;
    const second = await frame(pass, present);
    expect(second.stats?.historyReset).toBe(false);

    pass.view = 'split';
    pass.split = 0.5;
    const split = await frame(pass, present);
    // Left of the split shows NR off (identical to the off frame), right shows NR on.
    expect(pixel(split.pixels, 4, 40)).toEqual(pixel(off.pixels, 4, 40));
    expect(pixel(split.pixels, 60, 40)[0]).toBeGreaterThan(pixel(off.pixels, 60, 40)[0] + 10);

    pass.resetHistory();
    expect((await frame(pass, present)).stats?.historyReset).toBe(true);

    // NR off breaks the temporal chain: the next NR frame starts over.
    pass.view = 'off';
    await frame(pass, present);
    pass.view = 'on';
    expect((await frame(pass, present)).stats?.historyReset).toBe(true);

    await pass.setSize(80, 48);
    expect(built).toHaveLength(2);
    expect(built[0].disposed).toBe(true);
    expect(built[1].geometry.validWidth).toBe(80);
    pass.dispose();
    expect(built[1].disposed).toBe(true);
    present.dispose();
  });

  it('draws an external frame (a backend with its own frame, e.g. ReferenceFrame)', async () => {
    const { scene, camera } = testScene();
    const pass = new DlssNrPass({ renderer: gpu.renderer, scene, camera, width: WIDTH, height: HEIGHT, view: 'on' });
    const present = new RenderTarget(WIDTH, HEIGHT, { type: UnsignedByteType });
    const green = new Uint8Array(WIDTH * HEIGHT * 4);
    for (let i = 0; i < WIDTH * HEIGHT; ++i) green.set([0, 200, 0, 255], i * 4);
    const output = new DataTexture(green, WIDTH, HEIGHT);
    output.needsUpdate = true;
    const calls: { reset?: boolean; hasVelocity: boolean }[] = [];
    const external: DlssNrExternalFrame = {
      backend: stubBackend(gpu.renderer, WIDTH, HEIGHT),
      output,
      paperWhite: 1,
      colorStrength: 1,
      async render(input) {
        calls.push({ reset: input.reset, hasVelocity: input.velocity !== undefined });
        return { method: 'submitted-work-done', gpuMilliseconds: null, wallMilliseconds: 0 };
      },
      resetHistory() {},
      dispose() {},
    };
    await pass.setNetwork(async () => external);
    const first = await frame(pass, present);
    await frame(pass, present);
    expect(calls).toEqual([
      { reset: true, hasVelocity: true },
      { reset: false, hasVelocity: true },
    ]);
    expect(pixel(first.pixels, 10, 10)).toEqual([0, 200, 0]);
    pass.dispose();
    present.dispose();
  });
});
