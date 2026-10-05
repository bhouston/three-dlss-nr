import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { expect, it, vi } from 'vitest';
import { DEFAULT_MODEL_ID, HEAD_MODELS } from './models';
import { modelFitDistance } from './modelFraming';
import { loadHead } from './studio';

it('validates pinned sizes/hashes, embedded resources and glTF structure with Khronos validator', () => {
  const output = execFileSync(
    process.execPath,
    [new URL('../../scripts/validate-demo-models.mjs', import.meta.url).pathname],
    { encoding: 'utf8' },
  );
  expect(output.match(/0 errors/g)).toHaveLength(4);
});

it('keeps the original head as default', () => {
  expect(DEFAULT_MODEL_ID).toBe('lee-perry-smith');
  expect(HEAD_MODELS[0].id).toBe(DEFAULT_MODEL_ID);
  expect(HEAD_MODELS[0].loader?.height).toBe(2);
});

for (const entry of HEAD_MODELS.filter((model) => model.presentation)) {
  it(`normalizes and frames the actual ${entry.id} mesh while preserving embedded PBR materials`, async () => {
    const file = new URL(`../../public${entry.url}`, import.meta.url);
    const bytes = await readFile(file);
    const loader = new GLTFLoader();
    // Decode actual geometry/skins/materials in Node. Browser image decoding is tested separately.
    loader.register((parser: any) => {
      parser.loadTextureImage = () => Promise.resolve(new THREE.Texture());
      return { name: 'test-textures' };
    });
    const gltf = await loader.parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
    const before = new Map();
    gltf.scene.traverse((child: any) => {
      if (child.isMesh) before.set(child, child.material);
    });
    const spy = vi.spyOn(GLTFLoader.prototype, 'loadAsync').mockResolvedValue(gltf);
    const loaded = await loadHead(entry);
    try {
      for (const [mesh, material] of before) expect(mesh.material).toBe(material);
      expect(before.size).toBeGreaterThan(0);
      const box = new THREE.Box3().setFromObject(loaded.object);
      const size = box.getSize(new THREE.Vector3());
      expect(Math.max(size.x, size.y, size.z)).toBeCloseTo(2.4, 4);
      expect(box.getCenter(new THREE.Vector3()).length()).toBeLessThan(1e-5);
      for (const aspect of [16 / 9, 1, 0.4]) {
        const distance = modelFitDistance(loaded.radius, 26, aspect);
        const camera = new THREE.PerspectiveCamera(26, aspect, 0.1, 100);
        camera.position.copy(
          new THREE.Vector3(...entry.presentation!.cameraDirection).normalize().multiplyScalar(distance),
        );
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld(true);
        for (const x of [box.min.x, box.max.x])
          for (const y of [box.min.y, box.max.y])
            for (const z of [box.min.z, box.max.z]) {
              const projected = new THREE.Vector3(x, y, z).project(camera);
              expect(Math.abs(projected.x)).toBeLessThan(1);
              expect(Math.abs(projected.y)).toBeLessThan(1);
              expect(projected.z).toBeGreaterThan(-1);
              expect(projected.z).toBeLessThan(1);
            }
      }
    } finally {
      loaded.dispose();
      spy.mockRestore();
    }
  });
}
