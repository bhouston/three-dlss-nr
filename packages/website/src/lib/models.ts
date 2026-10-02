// The head models the demo can show. Data only: add an entry (and its files under public/models/<id>/) to offer
// another head in the model switcher. Every entry carries the attribution its license requires; the demo always
// shows the attribution of the head on screen. See docs/suggested-assets.md for vetted candidates and the licensing
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
}

export interface HeadModel {
  id: string;
  label: string;
  /** URL of the GLB / glTF file. */
  url: string;
  loader?: ModelLoaderHints;
  attribution: ModelAttribution;
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
];

export const DEFAULT_MODEL_ID = 'lee-perry-smith';

export function headModel(id: string): HeadModel {
  const model = HEAD_MODELS.find((entry) => entry.id === id);
  if (!model) throw new Error(`unknown head model "${id}"`);
  return model;
}
