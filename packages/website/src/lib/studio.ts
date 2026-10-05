// The demo's studio: a dark backdrop, a key / fill / rim light rig plus a soft room environment, and the head models
// of the registry loaded into it.

import * as THREE from 'three/webgpu';
import { texture as textureNode } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import type { HeadModel } from './models';
import { ModelResources } from './modelResources';

export interface Studio {
  scene: any;
  /** The group the current head lives in. */
  stage: any;
  dispose(): void;
}

/** Background, lights and environment (no head yet). */
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
  stage.name = 'head stage';
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

export interface LoadedHead {
  object: any;
  dispose(): void;
}

/** Load a registry entry: the glTF scene, its maps, a skin material, scaled and centred at the origin. */
export async function loadHead(entry: HeadModel): Promise<LoadedHead> {
  const resources = new ModelResources();
  try {
    const hints = entry.loader ?? {};
    const base = entry.url.slice(0, entry.url.lastIndexOf('/') + 1);
    const textureLoader = new THREE.TextureLoader();
    const loadMap = async (file: string | undefined, colorSpace: string) => {
      if (!file) return null;
      const map = await textureLoader.loadAsync(base + file);
      resources.texture(map);
      map.flipY = hints.flipY ?? false;
      map.colorSpace = colorSpace;
      map.anisotropy = 8;
      return map;
    };
    const [gltf, map, normalMap, specularMap] = await Promise.all([
      new GLTFLoader().loadAsync(entry.url).then((loaded: any) => {
        resources.gltf(loaded);
        return loaded;
      }),
      loadMap(hints.textures?.map, THREE.SRGBColorSpace),
      loadMap(hints.textures?.normalMap, THREE.NoColorSpace),
      loadMap(hints.textures?.specularMap, THREE.NoColorSpace),
    ]);
    const root = gltf.scenes[hints.sceneIndex ?? 0] ?? gltf.scene;

    if (hints.textures) {
      const material = new THREE.MeshPhysicalNodeMaterial({
        color: 0xffffff,
        map,
        normalMap,
        roughness: hints.roughness ?? 0.55,
        metalness: 0,
      });
      resources.material(material);
      if (normalMap) material.normalScale.set(hints.normalScale ?? 1, hints.normalScale ?? 1);
      if (specularMap) material.specularIntensityNode = textureNode(specularMap).r.mul(1.5);
      material.sheen = 0.15;
      material.sheenRoughness = 0.6;
      material.sheenColor = new THREE.Color(0.9, 0.7, 0.6);
      root.traverse((child: any) => {
        if (child.isMesh) child.material = material;
      });
    }

    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const scale = (hints.height ?? 2) / Math.max(size.y, 1e-6);
    const object = new THREE.Group();
    object.name = entry.id;
    root.position.sub(centre);
    object.add(root);
    object.scale.setScalar(scale);
    object.rotation.y = hints.rotationY ?? 0;

    return {
      object,
      dispose: () => resources.dispose(),
    };
  } catch (error) {
    resources.dispose();
    throw error;
  }
}
