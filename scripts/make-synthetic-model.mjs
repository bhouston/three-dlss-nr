// Write the deterministic synthetic model directory to disk:
//
//   pnpm build && node scripts/make-synthetic-model.mjs <dir> [--seed N]
//
// produces <dir>/manifest.json and <dir>/model/stages/s00.bin ... s10.bin (~141 MiB), the model directory layout
// of OpenDLSS-NR by maan (MIT, https://github.com/maanHimself/OpenDLSS-NR, docs/weights.md), which both the reference
// WebGPU port and three-dlss-nr load. These are synthetic weights for tests and the demo; no NVIDIA weights are
// involved. The bytes are identical on every machine (the stage SHA-256s are pinned in the library's tests).

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const generator = join(root, 'packages/three-dlss-nr/dist/synthetic/generate.js');

const args = process.argv.slice(2);
const seedAt = args.indexOf('--seed');
const seed = seedAt >= 0 ? Number(args.splice(seedAt, 2)[1]) : 1;
const [directory] = args;
if (!directory || !Number.isSafeInteger(seed)) {
  console.error('usage: node scripts/make-synthetic-model.mjs <dir> [--seed N]   (run `pnpm build` first)');
  process.exit(2);
}

let generateSyntheticModel;
try {
  ({ generateSyntheticModel } = await import(pathToFileURL(generator).href));
} catch (error) {
  console.error(`cannot load ${generator}: run \`pnpm build\` first (${error.message})`);
  process.exit(1);
}

const started = performance.now();
const { manifest, files } = await generateSyntheticModel({ seed });
let bytes = 0;
for (const [path, data] of files) {
  const target = join(directory, ...path.split('/'));
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, data);
  bytes += data.byteLength;
}
const seconds = ((performance.now() - started) / 1000).toFixed(1);
console.log(
  `wrote ${files.size} files (${(bytes / 2 ** 20).toFixed(1)} MiB, ${manifest.stages.length} stages, ` +
    `${manifest.tensors.length} records, seed ${seed}) to ${resolve(directory)} in ${seconds} s`,
);
