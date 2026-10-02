// Where the network's weights come from. None are hosted: the visitor picks a model directory on their own disk (read
// by the browser, never uploaded), or generates synthetic weights for testing.

import type { NRModelFilesLike } from 'three-dlss-nr';

export interface WeightsSource {
  kind: 'synthetic' | 'directory';
  /** Shown in the UI. */
  label: string;
  /** The model directory in memory: `manifest.json` and `model/<stage file>`. */
  model: NRModelFilesLike;
  /** Total bytes of the stage files. */
  bytes: number;
}

interface ManifestShape {
  totals?: { blockCount?: number };
  stages?: { id: string; file: string; packedByteLength: number }[];
  tensors?: unknown[];
}

const pathOf = (file: File) => (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;

/**
 * Read a model directory chosen with `<input type="file" webkitdirectory>`: find `manifest.json` (at any depth, the
 * shallowest wins), then the stage files it lists under `model/`. Nothing leaves the browser.
 */
export async function readModelDirectory(
  files: FileList | readonly File[],
  onProgress?: (message: string) => void,
): Promise<WeightsSource> {
  const list = Array.from(files);
  const depth = (file: File) => pathOf(file).split('/').length;
  const manifestFile = list
    .filter((file) => pathOf(file).split('/').at(-1) === 'manifest.json')
    .reduce<File | undefined>((best, file) => (!best || depth(file) < depth(best) ? file : best), undefined);
  if (!manifestFile) {
    throw new Error(
      'no manifest.json in the chosen folder. Pick the model directory itself: it holds manifest.json and a model/ ' +
        'folder with the stage files (model/stages/*.bin).',
    );
  }
  const root = pathOf(manifestFile).slice(0, -'manifest.json'.length);
  const manifestBytes = new Uint8Array(await manifestFile.arrayBuffer());
  let manifest: ManifestShape;
  try {
    manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as ManifestShape;
  } catch {
    throw new Error('manifest.json is not valid JSON');
  }
  if (!Array.isArray(manifest.stages) || !Array.isArray(manifest.tensors) || !manifest.totals?.blockCount) {
    throw new Error('manifest.json does not describe an OpenDLSS-NR model (no totals / stages / tensors)');
  }
  const byPath = new Map(list.map((file) => [pathOf(file), file]));
  const out = new Map<string, Uint8Array>([['manifest.json', manifestBytes]]);
  let bytes = 0;
  for (const [index, stage] of manifest.stages.entries()) {
    const path = `model/${stage.file}`;
    const file = byPath.get(root + path);
    if (!file) throw new Error(`the model directory is missing ${path} (listed in manifest.json)`);
    if (file.size !== stage.packedByteLength) {
      throw new Error(`${path} is ${file.size} bytes; manifest.json says ${stage.packedByteLength}`);
    }
    onProgress?.(`reading ${path} (${index + 1}/${manifest.stages.length})`);
    out.set(path, new Uint8Array(await file.arrayBuffer()));
    bytes += file.size;
  }
  const name = root.replace(/\/$/, '').split('/').at(-1) || 'model directory';
  return {
    kind: 'directory',
    label: `${name} (${(bytes / 1048576).toFixed(0)} MiB, local)`,
    model: { files: out },
    bytes,
  };
}

/** Deterministic synthetic weights (three-dlss-nr/synthetic): the right layout, meaningless output. */
export async function syntheticWeights(): Promise<WeightsSource> {
  const { generateSyntheticModel } = await import('three-dlss-nr/synthetic');
  const model = await generateSyntheticModel({ seed: 1 });
  let bytes = 0;
  for (const [path, data] of model.files) if (path.startsWith('model/')) bytes += data.byteLength;
  return {
    kind: 'synthetic',
    label: `synthetic, seed 1 (${(bytes / 1048576).toFixed(0)} MiB, generated in the browser)`,
    model,
    bytes,
  };
}
