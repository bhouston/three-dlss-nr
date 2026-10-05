import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three/webgpu';

import { HEAD_MODELS } from './models';
import { loadHead } from './studio';

const loads = vi.hoisted(() => ({ gltf: vi.fn() }));
vi.mock('three/addons/loaders/GLTFLoader.js', () => ({
  GLTFLoader: class {
    loadAsync = loads.gltf;
  },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function gltfFixture() {
  const geometry = new THREE.BoxGeometry();
  const material = new THREE.MeshStandardMaterial();
  const scene = new THREE.Group();
  scene.add(new THREE.Mesh(geometry, material));
  const spies = [vi.spyOn(geometry, 'dispose'), vi.spyOn(material, 'dispose')];
  return { scene, scenes: [scene], spies };
}

afterEach(() => vi.restoreAllMocks());

describe('loadHead cleanup', () => {
  it('releases successful siblings and late glTF/maps when a Promise.all load rejects', async () => {
    const gltf = deferred<any>();
    const lateMap = deferred<any>();
    const failingMap = deferred<any>();
    const first = new THREE.Texture();
    const late = new THREE.Texture();
    const firstDispose = vi.spyOn(first, 'dispose');
    const lateDispose = vi.spyOn(late, 'dispose');
    loads.gltf.mockReturnValue(gltf.promise);
    vi.spyOn(THREE.TextureLoader.prototype, 'loadAsync')
      .mockResolvedValueOnce(first)
      .mockReturnValueOnce(failingMap.promise)
      .mockReturnValueOnce(lateMap.promise);
    const result = loadHead(HEAD_MODELS[0]);
    failingMap.reject(new Error('normal map failed'));
    await expect(result).rejects.toThrow('normal map failed');
    expect(firstDispose).toHaveBeenCalledTimes(1);
    const f = gltfFixture();
    gltf.resolve(f);
    lateMap.resolve(late);
    await Promise.all([gltf.promise, lateMap.promise]);
    await Promise.resolve();
    for (const spy of f.spies) expect(spy).toHaveBeenCalledTimes(1);
    expect(lateDispose).toHaveBeenCalledTimes(1);
  });

  it('releases loaded resources when post-load scene setup throws', async () => {
    const f = gltfFixture();
    loads.gltf.mockResolvedValue(f);
    vi.spyOn(THREE.Box3.prototype, 'setFromObject').mockImplementation(() => {
      throw new Error('bad bounds');
    });
    await expect(loadHead({ ...HEAD_MODELS[0], loader: {} })).rejects.toThrow('bad bounds');
    for (const spy of f.spies) expect(spy).toHaveBeenCalledTimes(1);
  });

  it('retains original materials before replacing them and preserves head fitting defaults', async () => {
    const f = gltfFixture();
    loads.gltf.mockResolvedValue(f);
    const texture = new THREE.Texture();
    const textureDispose = vi.spyOn(texture, 'dispose');
    vi.spyOn(THREE.TextureLoader.prototype, 'loadAsync').mockResolvedValue(texture);
    const head = await loadHead(HEAD_MODELS[0]);
    const replacement = f.scene.children[0].material;
    const replacementDispose = vi.spyOn(replacement, 'dispose');
    expect(head.object.name).toBe(HEAD_MODELS[0].id);
    expect(head.object.scale.y).toBe(2);
    expect(texture.flipY).toBe(true);
    expect(texture.anisotropy).toBe(8);
    expect(replacement.roughness).toBe(0.55);
    head.dispose();
    head.dispose();
    for (const spy of f.spies) expect(spy).toHaveBeenCalledTimes(1);
    expect(replacementDispose).toHaveBeenCalledTimes(1);
    expect(textureDispose).toHaveBeenCalledTimes(1);
  });
});
