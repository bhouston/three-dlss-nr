// DlssNrPass end to end with the native TSL network on the synthetic model, in Node (Dawn): a rendered scene through
// input features, the 451-dispatch network and compose, with no validation errors, a finite head, the temporal
// history and a presented image that differs from NR off. (A `network.*` file: it runs in the gpu-network project.)

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Color, Mesh, OrthographicCamera, PlaneGeometry, Scene, UnsignedByteType } from 'three';
import { MeshBasicNodeMaterial, RenderTarget } from 'three/webgpu';

import { createGpuTestContext, type GpuTestContext } from '../../test/gpu.js';
import { NRModel } from '../model/Model.js';
import { NRNetwork, tslBackend } from '../NRNetwork.js';
import { generateSyntheticModel } from '../synthetic/generate.js';
import { backendBuilder, DlssNrPass } from './DlssNrPass.js';

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

const [WIDTH, HEIGHT] = [64, 48];

describe('DlssNrPass with the TSL network', () => {
  it('renders, runs the network and composes, with history', async () => {
    const renderer = gpu.renderer;
    const device: GPUDevice = renderer.backend.device;
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 5;
    const scene = new Scene();
    scene.background = new Color(0.2, 0.2, 0.25);
    const quad = new Mesh(new PlaneGeometry(1, 1), new MeshBasicNodeMaterial({ color: new Color(0.8, 0.3, 0.1) }));
    scene.add(quad);
    const present = new RenderTarget(WIDTH, HEIGHT, { type: UnsignedByteType });

    const pass = new DlssNrPass({ renderer, scene, camera, width: WIDTH, height: HEIGHT, view: 'off' });
    await pass.setNetwork(backendBuilder(tslBackend, model));
    expect(pass.state).toBe('ready');
    const backend = pass.backend as NRNetwork;
    expect(backend.id).toBe('tsl');
    expect(backend.geometry.validWidth).toBe(WIDTH);

    const frame = async () => {
      renderer.setRenderTarget(present);
      device.pushErrorScope('validation');
      const stats = await pass.render();
      const error = await device.popErrorScope();
      renderer.setRenderTarget(null);
      expect(error?.message ?? null).toBeNull();
      const pixels: Uint8Array = await renderer.readRenderTargetPixelsAsync(present, 0, 0, WIDTH, HEIGHT);
      return { stats, pixels: new Uint8Array(pixels.buffer, pixels.byteOffset, WIDTH * HEIGHT * 4) };
    };

    const off = await frame();
    pass.view = 'on';
    const first = await frame();
    expect(first.stats?.ran).toBe(true);
    expect(first.stats?.historyReset).toBe(true);
    const head = await backend.readHead();
    expect(head.every(Number.isFinite)).toBe(true);
    quad.position.x += 0.05;
    const second = await frame();
    expect(second.stats?.historyReset).toBe(false);
    expect(second.stats?.network?.wallMilliseconds).toBeGreaterThan(0);
    // Synthetic weights change the picture (meaninglessly): NR on differs from NR off.
    let differing = 0;
    for (let i = 0; i < off.pixels.length; ++i) if (off.pixels[i] !== second.pixels[i]) differing += 1;
    expect(differing).toBeGreaterThan(off.pixels.length / 10);

    pass.dispose();
    present.dispose();
  });
});
