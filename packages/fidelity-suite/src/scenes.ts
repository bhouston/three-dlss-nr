// The parity suite's three.js scenes: the Lee Perry-Smith head scan from two camera angles, and two procedural
// scenes with no external assets.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The head setup mirrors the website demo's studio
// (packages/website/src/lib/studio.ts) but is a copy on purpose: the committed results must not change when the demo
// is restyled. Head: "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities), CC BY 3.0, files served from
// packages/website/public/models/lee-perry-smith.

import * as THREE from 'three/webgpu';
import { texture as textureNode } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export interface SuiteScene {
  id: string;
  title: string;
  tags: string[];
  /** One paragraph for the scene README. */
  description: string;
  /** Build the scene; `modelsBase` is the URL of the website's `models/` directory. */
  create(renderer: any, modelsBase: string, aspect: number): Promise<{ scene: any; camera: any; dispose(): void }>;
}

/** The demo studio: dark backdrop, key / fill / rim lights, a soft room environment. */
function studio(renderer: any) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161b);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.04).texture;
  scene.environment = environment;
  scene.environmentIntensity = 0.35;
  const light = (color: [number, number, number], intensity: number, position: [number, number, number]) => {
    const l = new THREE.DirectionalLight(new THREE.Color(...color), intensity);
    l.position.set(...position);
    scene.add(l);
  };
  light([1.0, 0.94, 0.86], 3.2, [-3, 3.5, 4]);
  light([0.75, 0.85, 1.0], 0.7, [4, 0.5, 3]);
  light([0.85, 0.9, 1.0], 4.0, [2.5, 3, -4.5]);
  light([1.0, 0.9, 0.8], 1.5, [-3.5, 1, -3.5]);
  return {
    scene,
    dispose() {
      environment.dispose();
      pmrem.dispose();
    },
  };
}

async function loadHead(modelsBase: string) {
  const base = `${modelsBase}lee-perry-smith/`;
  const loader = new THREE.TextureLoader();
  const map = async (file: string, colorSpace: string) => {
    const t = await loader.loadAsync(base + file);
    t.flipY = true; // the maps use the image-top-down UV convention of the original three.js example
    t.colorSpace = colorSpace;
    t.anisotropy = 8;
    return t;
  };
  const [gltf, color, normal, specular] = await Promise.all([
    new GLTFLoader().loadAsync(`${base}LeePerrySmith.glb`),
    map('Map-COL.jpg', THREE.SRGBColorSpace),
    map('Infinite-Level_02_Tangent_SmoothUV.jpg', THREE.NoColorSpace),
    map('Map-SPEC.jpg', THREE.NoColorSpace),
  ]);
  const root = gltf.scenes[0];
  const material = new THREE.MeshPhysicalNodeMaterial({ color: 0xffffff, map: color, normalMap: normal });
  material.roughness = 0.55;
  material.metalness = 0;
  material.normalScale.set(0.8, 0.8);
  material.specularIntensityNode = textureNode(specular).r.mul(1.5);
  material.sheen = 0.15;
  material.sheenRoughness = 0.6;
  material.sheenColor = new THREE.Color(0.9, 0.7, 0.6);
  root.traverse((child: any) => {
    if (child.isMesh) child.material = material;
  });
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  root.position.sub(box.getCenter(new THREE.Vector3()));
  const head = new THREE.Group();
  head.add(root);
  head.scale.setScalar(2 / size.y);
  return {
    head,
    dispose() {
      root.traverse((child: any) => child.isMesh && child.geometry.dispose());
      material.dispose();
      for (const t of [color, normal, specular]) t.dispose();
    },
  };
}

function headScene(id: string, title: string, view: string, position: [number, number, number]): SuiteScene {
  return {
    id,
    title,
    tags: ['lee-perry-smith', 'head', view],
    description:
      `The Lee Perry-Smith head scan (${view} view) in the demo's studio: physically based skin with colour, normal ` +
      'and specular maps, four directional lights and a dim room environment, rendered by three.js WebGPURenderer ' +
      'into an RGBA16F target.',
    async create(renderer, modelsBase, aspect) {
      const room = studio(renderer);
      const { head, dispose } = await loadHead(modelsBase);
      room.scene.add(head);
      const camera = new THREE.PerspectiveCamera(26, aspect, 0.1, 100);
      camera.position.set(...position);
      camera.lookAt(0, 0.05, 0);
      return {
        scene: room.scene,
        camera,
        dispose() {
          dispose();
          room.dispose();
        },
      };
    },
  };
}

const spheres: SuiteScene = {
  id: 'spheres',
  title: 'Spheres and an emissive ring',
  tags: ['procedural', 'pbr', 'hdr'],
  description:
    'Twelve PBR spheres and boxes (rough to glossy, dielectric and metal) under a hemisphere and a directional ' +
    'light, with an emissive ring at intensity 6 whose HDR highlights exercise the display proxy shoulder.',
  async create(_renderer, _models, aspect) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x203040);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404060, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.set(3, 5, 2);
    scene.add(sun);
    const materials = [0xff6040, 0x40c0ff, 0xfff0a0, 0x80ff80].map(
      (color, i) => new THREE.MeshStandardMaterial({ color, roughness: 0.2 + i * 0.2, metalness: i % 2 ? 0.8 : 0.1 }),
    );
    const geometries = [new THREE.SphereGeometry(0.45, 48, 24), new THREE.BoxGeometry(0.7, 0.7, 0.7)];
    for (let i = 0; i < 12; ++i) {
      const mesh = new THREE.Mesh(geometries[i % 3 ? 0 : 1], materials[i % materials.length]);
      mesh.position.set((i % 4) * 1.2 - 1.8, Math.floor(i / 4) * 1.1 - 1.1, -((i * 7) % 5) * 0.6);
      mesh.rotation.set(i * 0.3, i * 0.7, 0);
      scene.add(mesh);
    }
    const ring = new THREE.TorusGeometry(0.6, 0.12, 24, 96);
    const glow = new THREE.MeshStandardMaterial({ color: 0x000000, emissive: 0xffa040, emissiveIntensity: 6 });
    const emissive = new THREE.Mesh(ring, glow);
    emissive.position.set(0, 0, 1);
    emissive.rotation.y = 0.4;
    scene.add(emissive);
    const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
    camera.position.set(0.3, 0.3, 5.6);
    camera.lookAt(0, 0, 0);
    return {
      scene,
      camera,
      dispose() {
        for (const m of [...materials, glow]) m.dispose();
        for (const g of [...geometries, ring]) g.dispose();
      },
    };
  },
};

const checkerRoom: SuiteScene = {
  id: 'checker-room',
  title: 'Checker room with thin lines',
  tags: ['procedural', 'texture', 'high-frequency'],
  description:
    'A floor and two walls with a nearest-filtered 16x16 checker texture, thin bright rods and a small cube under ' +
    'a point light: high-frequency texture and sub-pixel lines.',
  async create(_renderer, _models, aspect) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    const cells = 16;
    const pixels = new Uint8Array(cells * cells * 4);
    for (let y = 0; y < cells; ++y) {
      for (let x = 0; x < cells; ++x) {
        const on = (x + y) & 1;
        pixels.set(on ? [230, 225, 210, 255] : [40, 60, 90, 255], (y * cells + x) * 4);
      }
    }
    const checker = new THREE.DataTexture(pixels, cells, cells);
    checker.magFilter = THREE.NearestFilter;
    checker.minFilter = THREE.NearestFilter;
    checker.colorSpace = THREE.SRGBColorSpace;
    checker.needsUpdate = true;
    const surface = new THREE.MeshStandardMaterial({ map: checker, roughness: 0.8 });
    const plane = new THREE.PlaneGeometry(6, 6);
    const add = (rotation: [number, number, number], position: [number, number, number]) => {
      const mesh = new THREE.Mesh(plane, surface);
      mesh.rotation.set(...rotation);
      mesh.position.set(...position);
      scene.add(mesh);
    };
    add([-Math.PI / 2, 0, 0], [0, -1.5, 0]);
    add([0, 0, 0], [0, 1.5, -3]);
    add([0, Math.PI / 2, 0], [-3, 1.5, 0]);
    const rodMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x80c0ff, emissiveIntensity: 2 });
    const rod = new THREE.CylinderGeometry(0.012, 0.012, 4, 8);
    for (let i = 0; i < 7; ++i) {
      const mesh = new THREE.Mesh(rod, rodMaterial);
      mesh.position.set(-1.8 + i * 0.6, 0.2, -1 + (i % 2) * 0.5);
      mesh.rotation.z = 0.15 * (i - 3);
      scene.add(mesh);
    }
    const cubeMaterial = new THREE.MeshStandardMaterial({ color: 0xc04030, roughness: 0.3 });
    const box = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const cube = new THREE.Mesh(box, cubeMaterial);
    cube.position.set(0.8, -1.1, 0.4);
    cube.rotation.y = 0.6;
    scene.add(cube);
    const lamp = new THREE.PointLight(0xffeedd, 30, 0, 2);
    lamp.position.set(1, 2, 2);
    scene.add(lamp, new THREE.AmbientLight(0x404050, 0.6));
    const camera = new THREE.PerspectiveCamera(55, aspect, 0.1, 100);
    camera.position.set(1.6, 0.6, 4.2);
    camera.lookAt(-0.3, -0.2, -1);
    return {
      scene,
      camera,
      dispose() {
        for (const m of [surface, rodMaterial, cubeMaterial]) m.dispose();
        for (const g of [plane, rod, box]) g.dispose();
        checker.dispose();
      },
    };
  },
};

export const SCENES: readonly SuiteScene[] = [
  headScene('head-front', 'Lee Perry-Smith head, front', 'front', [0.3, 0.15, 5.3]),
  headScene('head-three-quarter', 'Lee Perry-Smith head, three-quarter', 'three-quarter', [3.6, 0.9, 4.9]),
  spheres,
  checkerRoom,
];
