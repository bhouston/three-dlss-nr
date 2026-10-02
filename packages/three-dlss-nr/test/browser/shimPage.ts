// Browser page for the reference-wgsl backend: parity against the standalone reference, and the backend benchmark.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Bundled by test/browser/chrome.mjs and run
// in a real Chrome (shader-f16 on D3D12 + DXC). Exposes `globalThis.nrShim = { parity, bench }`.
//
//   parity: the shim (vendored upstream code on three's GPUDevice, GPU-resident I/O) against the upstream port
//           running standalone (its own device from its own `requestDevice`, WGSL fetched from the submodule, the
//           upstream `Network.create`), on the same weights:
//             1. network: the same features -> head (bitwise) and every captured boundary;
//             2. frame: a three.js scene rendered into an RGBA16F MRT target (colour + velocity), two frames with a
//                moving camera. The shim reads the target on the GPU (ReferenceFrame); the standalone side gets the
//                same frame the way the upstream demo does - read back to the CPU and uploaded - with the velocity
//                converted on the CPU. Compared: packed inputs, features (noise lanes included: same GPU, same
//                compiler), head, presented image and history, per frame.
//   bench:  min / median ms per frame for each available backend and size.

import { Network } from '@ref/network.js';
import { HalfFloatType, PerspectiveCamera, RenderTarget, WebGPURenderer } from 'three/webgpu';
import { mrt, output, velocity } from 'three/tsl';
import * as THREE from 'three/webgpu';

import * as library from '../../src/index.js';
import { BUFFER_USAGE, MAP_MODE_READ } from '../../src/backend/gpuFlags.js';
import { createNRDevice } from '../../src/device.js';
import type { NRBackend, NRBackendFactory, NRFrameTiming } from '../../src/backend/NRBackend.js';
import { summarizeMilliseconds } from '../../src/backend/timing.js';
import {
  frameParams,
  loadReferenceModel,
  ReferenceFrame,
  referenceWgslBackend,
} from '../../src/reference-backend/index.js';
import { halfBits, halfToNumber, testFeatures } from './testFeatures.js';

const log = (...args: unknown[]) => console.log(...args);

async function createRenderer(): Promise<any> {
  const { device } = await createNRDevice();
  const renderer = new WebGPURenderer({ device, antialias: false });
  await renderer.init();
  return renderer;
}

function firstDifference(a: Uint8Array, b: Uint8Array): { count: number; first: number } {
  let count = 0;
  let first = -1;
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; ++i) {
    if (a[i] !== b[i]) {
      if (first < 0) first = i;
      count += 1;
    }
  }
  return { count, first };
}

const bytesOf = (view: ArrayBufferView) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

// ---------------------------------------------------------------------------------------------------------------
// parity
// ---------------------------------------------------------------------------------------------------------------

interface Verdict {
  name: string;
  bytes: number;
  mismatches: number;
  first: number;
}

function verdict(name: string, ours: Uint8Array, theirs: Uint8Array): Verdict {
  const { count, first } = firstDifference(ours, theirs);
  const sizeMismatch = ours.length !== theirs.length ? Math.abs(ours.length - theirs.length) : 0;
  return { name, bytes: ours.length, mismatches: count + sizeMismatch, first };
}

/** Read a GPU texture (2D, one mip) to tightly packed rows, as the upstream demo's readback hands frames over. */
async function readTexture(device: GPUDevice, texture: GPUTexture, bytesPerTexel: number): Promise<Uint8Array> {
  const { width, height } = texture;
  const pitch = Math.ceil((width * bytesPerTexel) / 256) * 256;
  const buffer = device.createBuffer({
    size: pitch * height,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: pitch }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(MAP_MODE_READ);
  const raw = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(width * height * bytesPerTexel);
  for (let y = 0; y < height; ++y)
    out.set(raw.subarray(y * pitch, y * pitch + width * bytesPerTexel), y * width * bytesPerTexel);
  buffer.unmap();
  buffer.destroy();
  return out;
}

/** A small procedural three.js scene with an MRT (colour + velocity) RGBA16F target. */
function buildScene(width: number, height: number) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x203040);
  const camera = new PerspectiveCamera(50, width / height, 0.1, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404060, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.position.set(3, 5, 2);
  scene.add(sun);
  const materials = [0xff6040, 0x40c0ff, 0xfff0a0, 0x80ff80].map(
    (color, i) => new THREE.MeshStandardMaterial({ color, roughness: 0.2 + i * 0.2, metalness: i % 2 ? 0.8 : 0.1 }),
  );
  for (let i = 0; i < 12; ++i) {
    const mesh = new THREE.Mesh(
      i % 3 ? new THREE.SphereGeometry(0.45, 32, 16) : new THREE.BoxGeometry(0.7, 0.7, 0.7),
      materials[i % materials.length],
    );
    mesh.position.set((i % 4) * 1.2 - 1.8, Math.floor(i / 4) * 1.1 - 1.1, -((i * 7) % 5) * 0.6);
    mesh.rotation.set(i * 0.3, i * 0.7, 0);
    scene.add(mesh);
  }
  const emissive = new THREE.Mesh(
    new THREE.TorusGeometry(0.6, 0.12, 16, 64),
    new THREE.MeshStandardMaterial({ color: 0x000000, emissive: 0xffa040, emissiveIntensity: 6 }),
  );
  emissive.position.set(0, 0, 1);
  scene.add(emissive);
  const target = new RenderTarget(width, height, { count: 2, type: HalfFloatType, depthBuffer: true, samples: 0 });
  return { scene, camera, target, spinner: emissive };
}

/** The standalone reference with the upstream demo's frame (production-pipeline.js), fed from CPU arrays. */
async function standaloneFrameNetwork(modelUrl: string, shaderBase: URL, width: number, height: number) {
  const resources: any = {};
  const network = await Network.create({
    weights: modelUrl,
    width,
    height,
    shaderBase,
    extraShaders: [{ path: 'shaders/frame.wgsl', entryPoints: ['input_features', 'compose'] }],
    before: (net: any) => {
      const device: GPUDevice = net.device;
      const storage = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;
      resources.scene = device.createBuffer({ label: 'rendered frame', size: width * height * 8, usage: storage });
      resources.motion = device.createBuffer({ label: 'velocity', size: width * height * 4, usage: storage });
      resources.history = [0, 1].map((i) =>
        device.createBuffer({ label: `history ${i}`, size: width * height * 8, usage: storage }),
      );
      resources.image = device.createBuffer({
        label: 'presented image',
        size: Math.ceil((width * 4) / 256) * 256 * height,
        usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
      });
      const g = net.geometry;
      net.recorder.pass(
        'input_features',
        { 1: resources.scene, 2: resources.history[0], 4: resources.motion, 5: net.features.buffer },
        new Uint32Array(18),
        [Math.ceil(g.fullWidth / 8), Math.ceil(g.fullHeight / 8)],
        'input features',
      );
      resources.featuresPass = net.recorder.passes.at(-1);
    },
    after: (net: any) => {
      net.recorder.pass(
        'compose',
        {
          1: resources.scene,
          2: resources.history[0],
          3: net.graph.head.buffer,
          4: resources.motion,
          6: resources.history[1],
          7: resources.image,
        },
        new Uint32Array(18),
        [Math.ceil(width / 8), Math.ceil(height / 8)],
        'compose',
      );
      resources.composePass = net.recorder.passes.at(-1);
      net.recorder.copy(resources.history[1], resources.history[0], width * height * 8, 'history swap');
    },
  });
  return { network, resources };
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<Uint8Array> {
  const staging = device.createBuffer({ size, usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_MODE_READ);
  const copy = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}

async function parity({
  modelUrl,
  shaderBase,
  width = 128,
  height = 128,
}: {
  modelUrl: string;
  shaderBase: string;
  width?: number;
  height?: number;
}) {
  const base = new URL(shaderBase, location.href);
  const results: Verdict[] = [];
  const renderer = await createRenderer();
  const device: GPUDevice = renderer.backend.device;
  const adapter = (device as any).adapterInfo ?? {};
  log(`device: ${adapter.vendor ?? ''} ${adapter.architecture ?? ''} ${adapter.description ?? ''}`);

  // 1. Network: identical features -> identical head and boundaries.
  log('parity: loading weights');
  const shared = await loadReferenceModel(renderer, modelUrl);
  const shim = await referenceWgslBackend.create({
    renderer,
    model: shared as any,
    width,
    height,
    captureBoundaries: true,
  });
  const features = testFeatures(shim.geometry.fullWidth, shim.geometry.fullHeight);
  shim.writeFeatures(features);
  await shim.run();
  log('parity: shim network done; creating standalone');
  const standalone = await Network.create({
    weights: modelUrl,
    width,
    height,
    captureBoundaries: true,
    shaderBase: base,
  });
  standalone.writeFeatures(features);
  await standalone.run();
  results.push(verdict('network: head', bytesOf(await shim.readHead()), bytesOf(await standalone.readHead())));
  if (shim.boundaryNames.join() !== standalone.boundaryNames.join()) throw new Error('boundary names differ');
  let boundaryMismatches = 0;
  for (const name of shim.boundaryNames) {
    const v = verdict(`network: ${name}`, await shim.readBoundary(name), await standalone.readBoundary(name));
    if (v.mismatches) results.push(v);
    boundaryMismatches += v.mismatches;
  }
  results.push({
    name: `network: ${shim.boundaryNames.length} boundaries`,
    bytes: 0,
    mismatches: boundaryMismatches,
    first: -1,
  });
  shim.dispose();
  standalone.destroy();

  // 2. Frame: three render target on the GPU vs the demo's CPU hand-over.
  log('parity: network compared; creating frames');
  const frame = await ReferenceFrame.create({ renderer, model: shared as any, width, height });
  const reference = await standaloneFrameNetwork(modelUrl, base, width, height);
  const refDevice: GPUDevice = reference.network.device;
  const { scene, camera, target, spinner } = buildScene(width, height);
  const blendScale = reference.network.model.blendScale();
  target.textures[0].name = 'output';
  target.textures[1].name = 'velocity';
  const outputs = mrt({ output, velocity });
  outputs.setClearColor('velocity', new THREE.Color(0, 0, 0), 0);
  renderer.setMRT(outputs);
  for (let index = 0; index < 3; ++index) {
    camera.position.set(0.15 * index, 0.2 + 0.05 * index, 6 - 0.2 * index);
    camera.lookAt(0, 0, 0);
    spinner.rotation.y = index * 0.2;
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    log(`parity: frame ${index} rendered; running shim frame`);
    await frame.render({ color: target.textures[0], velocity: target.textures[1] });
    log(`parity: frame ${index} shim done`);

    // The upstream demo's path: the frame read back to the CPU, motion converted, uploaded, flip_y = 0.
    const colorTexture: GPUTexture = renderer.backend.get(target.textures[0]).texture;
    const velocityTexture: GPUTexture = renderer.backend.get(target.textures[1]).texture;
    const rgba16 = await readTexture(device, colorTexture, 8);
    const velocityHalves = new Uint16Array((await readTexture(device, velocityTexture, 8)).buffer);
    const halves = new Uint16Array(width * height * 2);
    for (let i = 0; i < width * height; ++i) {
      halves[i * 2] = halfBits(-halfToNumber(velocityHalves[i * 4]) * 0.5);
      halves[i * 2 + 1] = halfBits(halfToNumber(velocityHalves[i * 4 + 1]) * 0.5);
    }
    const words = frameParams(
      reference.network.geometry,
      { enabled: true, intensity: 1, localTone: 1, localStructure: 1, skinStructure: -1, autoMask: true, style: 0 },
      index > 0,
      index,
      1,
      1,
      blendScale,
    );
    const r = reference.resources;
    refDevice.queue.writeBuffer(reference.network.recorder.paramsBuffer, r.featuresPass.index * 256, words);
    refDevice.queue.writeBuffer(reference.network.recorder.paramsBuffer, r.composePass.index * 256, words);
    refDevice.queue.writeBuffer(r.scene, 0, rgba16);
    refDevice.queue.writeBuffer(r.motion, 0, halves);
    await reference.network.run();

    const packed = await frame.readPacked();
    results.push(verdict(`frame ${index}: packed colour`, packed.scene, rgba16));
    results.push(verdict(`frame ${index}: packed motion`, packed.motion, bytesOf(halves)));
    const ourMotion = new Uint16Array(packed.motion.buffer, packed.motion.byteOffset, packed.motion.byteLength / 2);
    for (let i = 0, shown = 0; i < halves.length && shown < 6; ++i) {
      if (ourMotion[i] === halves[i]) continue;
      shown += 1;
      const source = velocityHalves[(i >> 1) * 4 + (i & 1)];
      log(
        `motion[${i}]: gpu 0x${ourMotion[i].toString(16)} cpu 0x${halves[i].toString(16)} from 0x${source.toString(16)}`,
      );
    }
    const fullRows = reference.network.geometry.fullRows;
    results.push(
      verdict(
        `frame ${index}: input features`,
        await frame.backend.readBuffer(frame.backend.network.features.buffer, fullRows * 64),
        await readBack(refDevice, reference.network.features.buffer, fullRows * 64),
      ),
    );
    results.push(
      verdict(
        `frame ${index}: head`,
        bytesOf(await frame.backend.readHead()),
        bytesOf(await reference.network.readHead()),
      ),
    );
    results.push(
      verdict(
        `frame ${index}: history`,
        await frame.readHistory(),
        await readBack(refDevice, r.history[0], width * height * 8),
      ),
    );
    const pitch = Math.ceil((width * 4) / 256) * 256;
    const refImage = await readBack(refDevice, r.image, pitch * height);
    const ourImage = await frame.backend.readBuffer((frame as any).resources.image, pitch * height);
    results.push(verdict(`frame ${index}: presented image`, ourImage, refImage));
  }
  renderer.setMRT(null);
  frame.dispose();
  reference.network.destroy();
  shared.dispose();
  target.dispose();
  return { adapter: `${adapter.vendor ?? ''} ${adapter.architecture ?? ''}`.trim(), width, height, results };
}

// ---------------------------------------------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------------------------------------------

async function bench({
  modelUrl,
  sizes = [
    [512, 512],
    [1280, 720],
  ],
  warmup = 5,
  frames = 30,
  only,
}: {
  modelUrl: string;
  sizes?: [number, number][];
  warmup?: number;
  frames?: number;
  only?: string[];
}) {
  const renderer = await createRenderer();
  const device: GPUDevice = renderer.backend.device;
  const info = (device as any).adapterInfo ?? {};
  // The native TSL backend registers itself as `tslBackend` in the package index once chunk E lands.
  const tslBackend = Object.entries(library).find(([name]) => name === 'tslBackend')?.[1] as
    | NRBackendFactory
    | undefined;
  const factories = [tslBackend, referenceWgslBackend as NRBackendFactory].filter(
    (f): f is NRBackendFactory => !!f && (!only || only.includes(f.id)),
  );
  const rows: unknown[] = [];
  let referenceModel: Awaited<ReturnType<typeof loadReferenceModel>> | null = null;
  for (const factory of factories) {
    const reason = factory.unavailableReason(renderer);
    if (reason) {
      rows.push({ backend: factory.id, skipped: reason });
      continue;
    }
    let model: any = modelUrl;
    if (factory.id === 'reference-wgsl') {
      const started = performance.now();
      referenceModel ??= await loadReferenceModel(renderer, modelUrl);
      log(`reference weights loaded in ${(performance.now() - started).toFixed(0)} ms`);
      model = referenceModel;
    }
    for (const [width, height] of sizes) {
      const started = performance.now();
      const backend: NRBackend = await factory.create({ renderer, model, width, height });
      const createMilliseconds = performance.now() - started;
      backend.writeFeatures(testFeatures(backend.geometry.fullWidth, backend.geometry.fullHeight));
      const first = await backend.run();
      for (let i = 0; i < warmup; ++i) await backend.run({ timing: true });
      const timings: NRFrameTiming[] = [];
      for (let i = 0; i < frames; ++i) timings.push(await backend.run({ timing: true }));
      const head = await backend.readHead();
      const row = {
        backend: backend.id,
        size: `${width}x${height}`,
        field: `${backend.geometry.fullWidth}x${backend.geometry.fullHeight}`,
        dispatches: backend.dispatchCount,
        method: timings[0].method,
        gpu: summarizeMilliseconds(timings.map((t) => t.gpuMilliseconds)),
        wall: summarizeMilliseconds(timings.map((t) => t.wallMilliseconds)),
        createMilliseconds,
        firstFrameMilliseconds: first.wallMilliseconds,
        activationMiB: backend.memory.activationBytes / 2 ** 20,
        weightMiB: backend.memory.weightBytes / 2 ** 20,
        headFinite: head.every(Number.isFinite),
      };
      log(JSON.stringify(row));
      rows.push(row);
      backend.dispose();
    }
  }
  referenceModel?.dispose();
  return { adapter: `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim(), rows };
}

/** Reject as soon as any WebGPU device is lost (e.g. a TDR on a busy GPU) instead of hanging the harness. */
function failOnDeviceLoss<A extends unknown[], R>(run: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
  return (...args: A) =>
    new Promise<R>((resolve, reject) => {
      const original = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
        const device = await original.call(this, descriptor);
        void device.lost.then((info) => {
          if (info.reason !== 'destroyed') reject(new Error(`WebGPU device lost: ${info.message}`));
        });
        return device;
      };
      run(...args).then(resolve, reject);
    });
}

(globalThis as any).nrShim = { parity: failOnDeviceLoss(parity), bench: failOnDeviceLoss(bench) };
