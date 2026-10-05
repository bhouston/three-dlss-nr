// The subjects the demo can show. Entries keep the historical HeadModel/HEAD_MODELS names for callers.
// Every entry carries attribution shown with the selected subject. See docs/suggested-assets.md for vetted candidates and the licensing
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
  /** Modifications made to our distributed copy. */
  modifications?: string;
}

/** How to turn the file into a lit head (glTF files without embedded textures need their maps assigned). */
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
  /** Turn the head to face the camera (radians about +y). */
  rotationY?: number;
  /** Height the head is scaled to, in scene units (default 2). */
  height?: number;
  /** Normalize the largest bound dimension instead of height (for wide or long subjects). */
  size?: number;
}

export interface HeadModel {
  id: string;
  label: string;
  /** URL of the GLB / glTF file. */
  url: string;
  loader?: ModelLoaderHints;
  attribution: ModelAttribution;
  /** Defaults for fitting this subject; embedded glTF cameras are not used. */
  presentation?: {
    cameraDirection: readonly [number, number, number];
    environmentIntensity: number;
  };
}

export const HEAD_MODELS: readonly HeadModel[] = [
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
    id: 'toy-car',
    label: 'Toy car (clearcoat and transmission)',
    url: '/models/toy-car/ToyCar.glb',
    loader: { size: 2.4 },
    presentation: { cameraDirection: [1.5, 0.65, 2], environmentIntensity: 1.1 },
    attribution: {
      title: 'Toy Car',
      author: 'Guido Odendahl and Eric Chadwick',
      license: 'CC0 1.0',
      licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl:
        'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/ToyCar',
      modifications: 'Unmodified upstream GLB.',
    },
  },
  {
    id: 'sheen-sofa',
    label: 'Sofa (fabric, leather and wood)',
    url: '/models/sheen-sofa/SheenWoodLeatherSofa.glb',
    loader: { size: 2.4 },
    presentation: { cameraDirection: [1, 0.45, 2.5], environmentIntensity: 0.65 },
    attribution: {
      title: 'Sheen Wood Leather Sofa',
      author: 'Fran Calvente; Eric Chadwick / Darmstadt Graphics Group GmbH',
      license: 'CC BY 4.0; original CC0 1.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl:
        'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/SheenWoodLeatherSofa',
      modifications: 'Unmodified upstream GLB.',
    },
  },
  {
    id: 'fox',
    label: 'Fox (static rigged pose)',
    url: '/models/fox/Fox.glb',
    loader: { size: 2.4 },
    presentation: { cameraDirection: [1, 0.3, 2], environmentIntensity: 0.35 },
    attribution: {
      title: 'Fox',
      author: 'PixelMannen; tomkranis; AsoboStudio and scurest',
      license: 'CC BY 4.0; original CC0 1.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl:
        'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/Fox',
      modifications: 'Unmodified upstream GLB.',
    },
  },
  {
    id: 'sci-fi-helmet',
    label: 'Sci-fi helmet (metal and paint)',
    url: '/models/sci-fi-helmet/SciFiHelmet.glb',
    loader: { size: 2.4 },
    presentation: { cameraDirection: [0.45, 0.15, 2], environmentIntensity: 1.0 },
    attribution: {
      title: 'SciFi Helmet',
      author: 'Michael Pavlovic and Norbert Nopper',
      license: 'CC0 1.0',
      licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
      source: 'Khronos glTF Sample Assets',
      sourceUrl:
        'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/SciFiHelmet',
      modifications: 'Packed as GLB; textures reduced to at most 1024 pixels. Geometry and materials unchanged.',
    },
  },
];

export const DEFAULT_MODEL_ID = 'lee-perry-smith';

export function headModel(id: string): HeadModel {
  const model = HEAD_MODELS.find((entry) => entry.id === id);
  if (!model) throw new Error(`unknown scene model "${id}"`);
  return model;
}
