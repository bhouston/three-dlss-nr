// The parity suite's fixed layout: renderers, outputs, where results live, and the README text generated from them.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Shared by generate.mjs and check.mjs.

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const suiteRoot = resolve(import.meta.dirname, '..');
export const resultsDir = join(suiteRoot, 'results');

export const REFERENCE = 'opendlss-nr';
export const RENDERERS = [
  { id: REFERENCE, label: 'OpenDLSS-NR (reference WebGPU port, standalone)', reference: true },
  { id: 'three-dlss-nr-tsl', label: 'three-dlss-nr: native TSL port' },
  { id: 'three-dlss-nr-shim', label: 'three-dlss-nr: reference WGSL shim' },
];
export const OTHERS = RENDERERS.filter((r) => !r.reference).map((r) => r.id);

export const OUTPUTS = [
  { id: 'output', label: 'Composed output' },
  { id: 'input', label: 'Input proxy' },
  { id: 'head-rgb', label: 'Head RGB residual' },
  { id: 'blend-logit', label: 'Blend weight' },
  { id: 'block-0', label: 'Block 0 (full field, 32 ch)' },
  { id: 'block-14', label: 'Block 14 (encoder 128)' },
  { id: 'vit-38', label: 'ViT output (block 38)' },
  { id: 'block-69', label: 'Block 69 (decoder 32)' },
];

export const FIDELITY_CONFIG = {
  title: 'three-dlss-nr parity: OpenDLSS-NR reference vs the three.js ports',
  renderers: RENDERERS,
  outputs: OUTPUTS,
};

/** At least the 79 boundaries, 3 post tensors, the head and the input features. */
export const MIN_TENSORS = 84;

/** Scene ids in display order (unlisted ones sort first, by name). */
export const SCENE_ORDER = ['head-front', 'head-three-quarter', 'spheres', 'checker-room'];

/** Pixels of a `<size>/<scene>` folder, for sorting. */
const pixels = (folder) =>
  folder
    .split('/')[0]
    .split('x')
    .map(Number)
    .reduce((a, b) => a * b, 1);

/** Position of a folder's scene in SCENE_ORDER. */
const order = (folder) => SCENE_ORDER.indexOf(folder.split('/')[1]);

/** Scene folders (`<size>/<scene>`) that have an exactness.json, sorted. */
export function sceneFolders() {
  if (!existsSync(resultsDir)) return [];
  const folders = [];
  for (const size of readdirSync(resultsDir, { withFileTypes: true })) {
    if (!size.isDirectory()) continue;
    for (const scene of readdirSync(join(resultsDir, size.name), { withFileTypes: true })) {
      if (scene.isDirectory() && existsSync(join(resultsDir, size.name, scene.name, 'exactness.json')))
        folders.push(`${size.name}/${scene.name}`);
    }
  }
  return folders.toSorted((a, b) => pixels(a) - pixels(b) || order(a) - order(b) || a.localeCompare(b));
}

export const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));

const ms = (value) => (value === null || value === undefined ? 'n/a' : value.toFixed(1));
const SHORT = { [REFERENCE]: 'reference', 'three-dlss-nr-tsl': 'TSL', 'three-dlss-nr-shim': 'shim' };

// The viewer renders headings, paragraphs, lists, emphasis and links, but no tables, so the READMEs use lists.

/** One scene's README: what it is, what was compared, and what came out. */
export function sceneReadme(exactness, timing) {
  const g = exactness.geometry;
  const lines = [
    exactness.description,
    '',
    `Valid size ${exactness.size}, padded field ${g.field}, pooled levels ${g.levels.join(', ')}, ${g.vitTokens} ViT ` +
      `tokens. Synthetic weights: the images are parity evidence, not pictures. Device: ${exactness.adapter}, Chrome ` +
      '(D3D12 + DXC, hardware f16).',
    '',
    '## Exactness against the standalone reference',
    '',
  ];
  for (const entry of RENDERERS.filter((candidate) => !candidate.reference)) {
    const s = exactness.summary[entry.id];
    lines.push(
      `- **${entry.label}**: ${s.verdict}. Raw tensors ${s.tensors} byte-identical, PNGs ${s.images} byte-identical.`,
    );
  }
  lines.push(
    '',
    `The ${exactness.tensors.length} tensors: the input features each renderer ran on, all ` +
      `${exactness.tensors.length - 5} captured block boundaries, the three post tensors and the f32 head ` +
      '(verdicts and SHA-256 digests in exactness.json).',
  );
  if (timing) {
    const base = timing.renderers[REFERENCE].gpu?.median;
    lines.push('', '## GPU time per frame', '');
    for (const r of RENDERERS) {
      const t = timing.renderers[r.id];
      const ratio = base && t.gpu ? `, ${(t.gpu.median / base).toFixed(2)}x the reference` : '';
      lines.push(
        `- **${r.label}**: median ${ms(t.gpu?.median)} ms (min ${ms(t.gpu?.min)}, wall ${ms(t.wall?.median)})${ratio}`,
      );
    }
    lines.push(
      '',
      `Whole network, ${timing.renderers[REFERENCE].dispatches} dispatches, no boundary captures; ` +
        `${timing.renderers[REFERENCE].frames} frames after ${timing.warmup} warm-up frames, renderers interleaved ` +
        'frame by frame, timestamp queries around the frame.',
    );
  }
  if (exactness.tags.includes('lee-perry-smith')) {
    lines.push(
      '',
      'Head: "Infinite, 3D Head Scan" by Lee Perry-Smith ([Infinite-Realities](https://ir-ltd.net)), ' +
        '[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via the three.js examples.',
    );
  }
  return `${lines.join('\n')}\n`;
}

/** The results preamble: what is compared and how, then a summary over every scene on disk. */
export function resultsReadme(scenes) {
  const rows = scenes.map(({ exactness, timing }) => {
    const exact = OTHERS.every((id) => exactness.summary[id].verdict === 'bit-exact');
    const times = RENDERERS.map((r) => `${SHORT[r.id]} ${ms(timing?.renderers[r.id]?.gpu?.median)}`).join(', ');
    return `- **${exactness.title}**: ${exact ? 'all bit-exact' : '**NOT bit-exact**'}; GPU ms per frame: ${times}`;
  });
  const adapter = scenes[0]?.exactness.adapter ?? 'n/a';
  return `Three renderers run each scene on the same input features and the same weights: **reference**, the
standalone WebGPU port of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
[maan](https://github.com/maanHimself) (MIT, pinned at 9d08f41); **TSL**, the native three.js port
([three-dlss-nr](https://github.com/bhouston/three-dlss-nr)); and **shim**, the reference's own WGSL running inside
three.js on the renderer's device.

**The weights are synthetic** (NVIDIA's weights are not included anywhere), so the outputs are parity evidence, not
pictures. **Bit-exactness is checked on the raw tensors**: each renderer's input features, all 79 block boundaries, the
post tensors and the f32 head are compared byte for byte with the reference (exactness.json in each scene folder). The
images are deterministic views of those tensors, so every pair shows an infinite PSNR, and CI fails if any verdict or
image hash regresses. How the scenes, features and visualizations are made:
[packages/fidelity-suite](https://github.com/bhouston/three-dlss-nr/tree/main/packages/fidelity-suite).

## Summary

GPU time per frame, median, in ms; ${adapter}, Chrome stable (D3D12 + DXC). The shim runs the reference's
WGSL, so it matches the reference's speed; the TSL port computes the same bytes with generated kernels and is about
1.6x slower.

${rows.join('\n')}
`;
}

/**
 * Rewrite every README from the exactness.json and timing.json files on disk: one per scene, one per size group and
 * the results preamble. Returns the files written.
 */
export function writeReadmes() {
  const scenes = sceneFolders().map((folder) => ({
    folder,
    exactness: readJson(join(resultsDir, folder, 'exactness.json')),
    timing: existsSync(join(resultsDir, folder, 'timing.json'))
      ? readJson(join(resultsDir, folder, 'timing.json'))
      : null,
  }));
  const files = [];
  const write = (path, text) => {
    writeFileSync(path, text);
    files.push(path);
  };
  for (const { folder, exactness, timing } of scenes)
    write(join(resultsDir, folder, 'README.md'), sceneReadme(exactness, timing));
  for (const size of new Set(scenes.map(({ folder }) => folder.split('/')[0]))) {
    const { geometry } = scenes.find(({ folder }) => folder.startsWith(`${size}/`)).exactness;
    write(
      join(resultsDir, size, 'README.md'),
      `Valid size ${size}: the network runs on a ${geometry.field} padded field, pooled to ` +
        `${geometry.levels.join(', ')}, with ${geometry.vitTokens} ViT tokens.\n`,
    );
  }
  write(join(resultsDir, 'README.md'), resultsReadme(scenes));
  return files;
}
