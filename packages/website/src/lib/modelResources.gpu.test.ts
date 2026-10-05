import { afterAll, beforeAll, expect, it } from 'vitest';
import { createCanvas } from 'vitest-environment-webgpu-node';
import * as THREE from 'three/webgpu';
import { createNRDevice, createNRRenderer } from 'three-dlss-nr';

import { ModelResources } from './modelResources';

let device: GPUDevice;
let renderer: any;

beforeAll(async () => {
  ({ device } = await createNRDevice());
  ({ renderer } = await createNRRenderer({ device, canvas: createCanvas(32, 32).asElement() }));
});

afterAll(() => {
  renderer?.dispose();
  device?.destroy();
});

it('returns renderer geometry/texture counters to baseline after repeated model swaps', async () => {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10);
  camera.position.z = 3;
  await renderer.renderAsync(scene, camera);
  const baseline = { ...renderer.info.memory };
  for (let cycle = 0; cycle < 3; cycle++) {
    const geometry = new THREE.BoxGeometry();
    const texture = new THREE.DataTexture(new Uint8Array([255, 128, 64, 255]), 1, 1);
    texture.needsUpdate = true;
    const material = new THREE.MeshBasicNodeMaterial({ map: texture });
    const model = new THREE.Group();
    model.add(new THREE.Mesh(geometry, material), new THREE.Mesh(geometry, material));
    const resources = new ModelResources();
    resources.gltf({ scene: model, scenes: [model] });
    scene.add(model);
    await renderer.renderAsync(scene, camera);
    await device.queue.onSubmittedWorkDone();
    expect(renderer.info.memory.geometries).toBeGreaterThan(baseline.geometries);
    expect(renderer.info.memory.textures).toBeGreaterThan(baseline.textures);
    scene.remove(model);
    resources.dispose();
    resources.dispose();
    expect(renderer.info.memory.geometries).toBe(baseline.geometries);
    expect(renderer.info.memory.textures).toBe(baseline.textures);
  }
});
