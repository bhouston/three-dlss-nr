// Models and scenes the demo can show. Add an entry and its files under public/models/<id>/ to offer another
// subject in the model switcher. Every entry carries the attribution its license requires; the demo always
// shows the attribution on screen. See docs/suggested-assets.md for vetted candidates and the licensing
// policy (any Creative Commons license except NC / ND).

export interface ModelAttribution {
  /** Title of the work, as its author published it. */
  title: string;
  titleUrl?: string;
  author: string;
  authorUrl?: string;
  /** License name, e.g. 'CC BY 3.0'. */
  license: string;
  licenseUrl: string;
  /** Where our copy came from (shown after the credit). */
  source?: string;
  sourceUrl?: string;
  /** Changes made to the downloaded asset and displayed frames. */
  modifications?: string;
}

/** Framing and optional external maps; embedded glTF PBR materials are preserved. */
export interface ModelLoaderHints {
  /** Which glTF scene to use (default 0; e.g. the Lee Perry-Smith GLB has a second, unused scene with a lamp). */
  sceneIndex?: number;
  /** Texture maps to assign to every mesh, relative to the model's directory. */
  textures?: {
    /** Base colour (sRGB). */
    map?: string;
    /** Tangent-space normal map. */
    normalMap?: string;
    /** Specular intensity (red channel). */
    specularMap?: string;
  };
  /** glTF UV convention for the external maps (default false: glTF textures are not flipped). */
  flipY?: boolean;
  normalScale?: number;
  roughness?: number;
  /** Turn the subject to face the camera (radians about +y). */
  rotationY?: number;
  /** Height the subject is scaled to, in scene units (default 2). */
  height?: number;
  /** Scale the longest bounding-box dimension to this size instead of fitting by height. */
  maxSize?: number;
  /** A useful starting angle for this subject, in the normalized studio coordinates. */
  cameraPosition?: readonly [number, number, number];
  environmentIntensity?: number;
}

export interface DemoModel {
  id: string;
  label: string;
  /** URL of the GLB / glTF file. */
  url: string;
  loader?: ModelLoaderHints;
  attribution: ModelAttribution;
}

export const DEMO_MODELS: readonly DemoModel[] = [
  {
    id: 'lee-perry-smith',
    label: 'Lee Perry-Smith (head scan)',
    url: '/models/lee-perry-smith/LeePerrySmith.glb',
    loader: {
      sceneIndex: 0,
      textures: {
        map: 'Map-COL.jpg',
        normalMap: 'Infinite-Level_02_Tangent_SmoothUV.jpg',
        specularMap: 'Map-SPEC.jpg',
      },
      // The maps were made for the original three.js JSON model, whose UVs use the image-top-down convention:
      // three.js's examples load them with TextureLoader's default (flipped) orientation, and so do we.
      flipY: true,
      normalScale: 0.8,
      roughness: 0.55,
      height: 2,
    },
    attribution: {
      title: 'Infinite, 3D Head Scan',
      author: 'Lee Perry-Smith (Infinite-Realities)',
      authorUrl: 'http://www.ir-ltd.net/',
      license: 'CC BY 3.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/3.0/',
      source: 'glTF via the three.js examples',
      sourceUrl: 'https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/LeePerrySmith',
    },
  },
  {
    id: 'car-concept',
    label: 'Sportautó (Car Concept)',
    url: '/models/car-concept/scene.glb',
    loader: { maxSize: 3.4, cameraPosition: [3.2, 1.6, 4.8], environmentIntensity: 0.85 },
    attribution: {
      title: 'Car Concept',
      titleUrl: 'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CarConcept',
      author: 'Eric Chadwick / Darmstadt Graphics Group',
      license: 'CC BY 4.0 (logos excluded)',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl: '/models/car-concept/ATTRIBUTION.txt',
      modifications: 'Textures resized and packed as GLB; NR on re-renders the image.',
    },
  },
  {
    id: 'leather-sofa',
    label: 'Bőrkanapé és mintás párnák',
    url: '/models/leather-sofa/scene.glb',
    loader: { maxSize: 3.2, cameraPosition: [1.8, 1.2, 5.5], environmentIntensity: 0.6 },
    attribution: {
      title: 'Sheen Wood Leather Sofa',
      titleUrl: 'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SheenWoodLeatherSofa',
      author: 'Fran Calvente; Eric Chadwick / Darmstadt Graphics Group',
      license: 'CC0 + CC BY 4.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl: '/models/leather-sofa/ATTRIBUTION.txt',
      modifications: 'Repacked as GLB; NR on re-renders the image.',
    },
  },
  {
    id: 'fox',
    label: 'Róka (stilizált állat)',
    url: '/models/fox/scene.glb',
    loader: { maxSize: 3.2, cameraPosition: [5, 1.4, 2.7], environmentIntensity: 0.5 },
    attribution: {
      title: 'Fox',
      titleUrl: 'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox',
      author: 'PixelMannen; tomkranis; AsoboStudio; scurest',
      license: 'CC0 + CC BY 4.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl: '/models/fox/ATTRIBUTION.txt',
      modifications: 'Textures re-encoded and packed as GLB; NR on re-renders the image.',
    },
  },
  {
    id: 'flight-helmet',
    label: 'Pilótasisak (bőr, fém, üveg)',
    url: '/models/flight-helmet/scene.glb',
    loader: { height: 2, cameraPosition: [1.2, 0.7, 5.4], environmentIntensity: 0.7 },
    attribution: {
      title: 'Flight Helmet',
      titleUrl: 'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/FlightHelmet',
      author: 'Gary Hsu (glTF conversion)',
      license: 'CC0',
      licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl: '/models/flight-helmet/ATTRIBUTION.txt',
      modifications: 'Textures resized and packed as GLB; NR on re-renders the image.',
    },
  },
];

export const DEFAULT_MODEL_ID = 'lee-perry-smith';

export function demoModel(id: string): DemoModel {
  const model = DEMO_MODELS.find((entry) => entry.id === id);
  if (!model) throw new Error(`unknown demo model "${id}"`);
  return model;
}
