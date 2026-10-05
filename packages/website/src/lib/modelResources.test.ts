import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three/webgpu';

import { ModelResources } from './modelResources';

class Bitmap {
  close = vi.fn();
}

const fixture = () => {
  const geometry = new THREE.BoxGeometry();
  const bitmap = new Bitmap();
  const texture = new THREE.Texture(bitmap);
  const clone = texture.clone();
  const original = new THREE.MeshStandardMaterial({ map: texture });
  const second = new THREE.MeshStandardMaterial({ normalMap: clone });
  const scene = new THREE.Group();
  scene.add(new THREE.Mesh(geometry, [original, second]), new THREE.Mesh(geometry, original));
  const unusedGeometry = new THREE.SphereGeometry();
  const unusedMaterial = new THREE.MeshStandardMaterial({ map: texture });
  const unused = new THREE.Group();
  unused.add(new THREE.Mesh(unusedGeometry, unusedMaterial));
  const disposables = [geometry, original, second, unusedGeometry, unusedMaterial, texture, clone];
  const spies = disposables.map((resource) => vi.spyOn(resource, 'dispose'));
  return { bitmap, texture, scene, gltf: { scene, scenes: [scene, unused] }, spies };
};

afterEach(() => {
  THREE.Cache.clear();
  THREE.Cache.enabled = false;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('model resource ownership', () => {
  it('releases shared geometry, multi-materials, unused scenes, replaced materials and shared bitmap sources once', () => {
    vi.stubGlobal('ImageBitmap', Bitmap);
    const f = fixture();
    const owned = new ModelResources();
    owned.gltf(f.gltf);
    const replacement = new THREE.MeshPhysicalNodeMaterial({ map: f.texture });
    const disposeReplacement = vi.spyOn(replacement, 'dispose');
    owned.material(replacement);
    f.scene.traverse((child: any) => {
      if (child.isMesh) child.material = replacement;
    });
    owned.dispose();
    owned.dispose();
    for (const spy of f.spies) expect(spy).toHaveBeenCalledTimes(1);
    expect(disposeReplacement).toHaveBeenCalledTimes(1);
    expect(f.bitmap.close).toHaveBeenCalledTimes(1);
  });

  it('keeps shared image sources alive until their last loaded owner is disposed', () => {
    vi.stubGlobal('ImageBitmap', Bitmap);
    const image = new Bitmap();
    const a = new ModelResources();
    const b = new ModelResources();
    a.texture(new THREE.Texture(image));
    b.texture(new THREE.Texture(image));
    a.dispose();
    expect(image.close).not.toHaveBeenCalled();
    b.dispose();
    expect(image.close).toHaveBeenCalledTimes(1);
  });

  it('does not close cached bitmaps or arbitrary borrowed image objects', () => {
    vi.stubGlobal('ImageBitmap', Bitmap);
    const cached = new Bitmap();
    const borrowed = { close: vi.fn() };
    THREE.Cache.enabled = true;
    THREE.Cache.add('image-bitmap:fixture', cached);
    const owned = new ModelResources();
    owned.texture(new THREE.Texture(cached));
    owned.texture(new THREE.Texture(borrowed));
    // Removing the cache entry does not transfer ownership to the model.
    THREE.Cache.clear();
    owned.dispose();
    expect(cached.close).not.toHaveBeenCalled();
    expect(borrowed.close).not.toHaveBeenCalled();
  });

  it('does not close an owned bitmap subsequently retained by the global cache', () => {
    vi.stubGlobal('ImageBitmap', Bitmap);
    const image = new Bitmap();
    const owned = new ModelResources();
    owned.texture(new THREE.Texture(image));
    THREE.Cache.enabled = true;
    THREE.Cache.add('image-bitmap:fixture', image);
    owned.dispose();
    expect(image.close).not.toHaveBeenCalled();
  });

  it('cleans late arrivals after disposal, including all scenes, without double disposal', () => {
    vi.stubGlobal('ImageBitmap', Bitmap);
    const f = fixture();
    const owned = new ModelResources();
    owned.dispose();
    owned.gltf(f.gltf);
    owned.gltf(f.gltf);
    owned.dispose();
    for (const spy of f.spies) expect(spy).toHaveBeenCalledTimes(1);
    expect(f.bitmap.close).toHaveBeenCalledTimes(1);
  });
});
