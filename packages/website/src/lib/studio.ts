// The demo's studio: a dark backdrop, a key / fill / rim light rig plus a soft room environment, and the models
// of the registry loaded into it.

import * as THREE from 'three/webgpu';
import { texture as textureNode } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import type { DemoModel } from './models';

export interface Studio {
  scene: any;
  /** The group the current model lives in. */
  stage: any;
  dispose(): void;
}

/** Background, lights and environment (no model yet). */
export function createStudio(renderer: any): Studio {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161b);

  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.04).texture;
  scene.environment = environment;
  scene.environmentIntensity = 0.35;

  // Key: warm, high camera-left. Fill: cool and dim, low camera-right. Rim: behind, from above right, to separate the
  // silhouette from the backdrop.
  const key = new THREE.DirectionalLight(new THREE.Color(1.0, 0.94, 0.86), 3.2);
  key.position.set(-3, 3.5, 4);
  const fill = new THREE.DirectionalLight(new THREE.Color(0.75, 0.85, 1.0), 0.7);
  fill.position.set(4, 0.5, 3);
  const rim = new THREE.DirectionalLight(new THREE.Color(0.85, 0.9, 1.0), 4.0);
  rim.position.set(2.5, 3, -4.5);
  const rimLeft = new THREE.DirectionalLight(new THREE.Color(1.0, 0.9, 0.8), 1.5);
  rimLeft.position.set(-3.5, 1, -3.5);
  scene.add(key, fill, rim, rimLeft);

  const stage = new THREE.Group();
  stage.name = 'model stage';
  scene.add(stage);

  return {
    scene,
    stage,
    dispose() {
      environment.dispose();
      pmrem.dispose();
      room.dispose?.();
    },
  };
}

export interface LoadedModel {
  object: any;
  dispose(): void;
}

/** Load a registry entry, preserving embedded PBR materials and fitting it to the studio. */
export async function loadModel(entry: DemoModel): Promise<LoadedModel> {
  const hints = entry.loader ?? {};
  const base = entry.url.slice(0, entry.url.lastIndexOf('/') + 1);
  const textureLoader = new THREE.TextureLoader();
  const loadMap = async (file: string | undefined, colorSpace: string) => {
    if (!file) return null;
    const map = await textureLoader.loadAsync(base + file);
    map.flipY = hints.flipY ?? false;
    map.colorSpace = colorSpace;
    map.anisotropy = 8;
    return map;
  };
  const [gltf, map, normalMap, specularMap] = await Promise.all([
    new GLTFLoader().loadAsync(entry.url),
    loadMap(hints.textures?.map, THREE.SRGBColorSpace),
    loadMap(hints.textures?.normalMap, THREE.NoColorSpace),
    loadMap(hints.textures?.specularMap, THREE.NoColorSpace),
  ]);
  const root = gltf.scenes[hints.sceneIndex ?? 0] ?? gltf.scene;

  const geometries = new Set<any>();
  const materials = new Set<any>();
  const textures = new Set<any>([map, normalMap, specularMap].filter(Boolean));
  const collectMaterial = (material: any) => {
    materials.add(material);
    for (const value of Object.values(material) as any[]) if (value?.isTexture) textures.add(value);
  };
  // Include shared resources and unused glTF scenes. A model switch releases each
  // geometry, material and embedded texture once, including multi-material meshes.
  for (const scene of gltf.scenes) {
    scene.traverse((child: any) => {
      if (!child.isMesh) return;
      geometries.add(child.geometry);
      for (const material of Array.isArray(child.material) ? child.material : [child.material]) {
        collectMaterial(material);
      }
    });
  }
  if (hints.textures) {
    const material = new THREE.MeshPhysicalNodeMaterial({
      color: 0xffffff,
      map,
      normalMap,
      roughness: hints.roughness ?? 0.55,
      metalness: 0,
    });
    if (normalMap) material.normalScale.set(hints.normalScale ?? 1, hints.normalScale ?? 1);
    if (specularMap) material.specularIntensityNode = textureNode(specularMap).r.mul(1.5);
    material.sheen = 0.15;
    material.sheenRoughness = 0.6;
    material.sheenColor = new THREE.Color(0.9, 0.7, 0.6);
    collectMaterial(material);
    root.traverse((child: any) => {
      if (child.isMesh) child.material = material;
    });
  }

  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const scale = hints.maxSize
    ? hints.maxSize / Math.max(size.x, size.y, size.z, 1e-6)
    : (hints.height ?? 2) / Math.max(size.y, 1e-6);
  const object = new THREE.Group();
  object.name = entry.id;
  root.position.sub(centre);
  object.add(root);
  object.scale.setScalar(scale);
  object.rotation.y = hints.rotationY ?? 0;

  return {
    object,
    dispose() {
      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      const images = new Set<any>();
      for (const texture of textures) {
        texture.dispose();
        if (texture.image) images.add(texture.image);
      }
      for (const image of images) image.close?.();
    },
  };
}
