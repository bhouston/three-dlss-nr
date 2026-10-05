// Resources returned by the model loader belong to its load, including unused glTF scenes and replaced materials.
import { Cache } from 'three/webgpu';

type Disposable = { dispose(): void };
// ImageBitmapLoader can share a cached source across independent glTF loads. Even if the cache is later cleared,
// a bitmap must stay open while another loaded model uses it. Cached images remain owned by their cache/caller.
const imageOwners = new WeakMap<ImageBitmap, { count: number; owned: boolean }>();
const cachedImage = (image: ImageBitmap): boolean => Object.values(Cache.files).includes(image);

function retainImage(image: ImageBitmap): void {
  const owner = imageOwners.get(image);
  if (owner) {
    owner.count += 1;
    owner.owned &&= !cachedImage(image);
  } else imageOwners.set(image, { count: 1, owned: !cachedImage(image) });
}

function releaseImage(image: ImageBitmap): void {
  const owner = imageOwners.get(image)!;
  if (--owner.count !== 0) return;
  imageOwners.delete(image);
  if (owner.owned && !cachedImage(image)) image.close();
}

/** A load can fail before all requests finish. Late arrivals are immediately released after disposal. */
export class ModelResources {
  private geometries = new Set<Disposable>();
  private materials = new Set<Disposable>();
  private textures = new Set<Disposable>();
  private images = new Set<ImageBitmap>();
  private disposed = false;

  private add(set: Set<Disposable>, resource: Disposable): void {
    if (set.has(resource)) return;
    set.add(resource);
    if (this.disposed) resource.dispose();
  }

  texture(texture: any): void {
    if (!texture?.isTexture || this.textures.has(texture)) return;
    // Texture.dispose() does not close its source. Only actual ImageBitmaps are closable here; DOM images,
    // canvases and arbitrary user objects are borrowed. Retain sources before releasing late texture arrivals.
    const images = Array.isArray(texture.image) ? texture.image : [texture.image];
    const added: ImageBitmap[] = [];
    for (const image of images) {
      if (typeof ImageBitmap === 'undefined' || !(image instanceof ImageBitmap) || this.images.has(image)) continue;
      this.images.add(image);
      retainImage(image);
      added.push(image);
    }
    this.add(this.textures, texture);
    if (this.disposed) for (const image of added) releaseImage(image);
  }

  material(material: any): void {
    if (!material || this.materials.has(material)) return;
    for (const value of Object.values(material)) this.texture(value);
    this.add(this.materials, material);
  }

  gltf(gltf: any): void {
    for (const scene of new Set([gltf.scene, ...(gltf.scenes ?? [])])) {
      scene?.traverse((child: any) => {
        if (child.geometry) this.add(this.geometries, child.geometry);
        for (const material of Array.isArray(child.material) ? child.material : [child.material]) {
          this.material(material);
        }
      });
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const set of [this.geometries, this.materials, this.textures]) {
      for (const resource of set) resource.dispose();
    }
    for (const image of this.images) releaseImage(image);
  }
}
