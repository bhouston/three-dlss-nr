#!/usr/bin/env node
// Check the committed parity results: fail unless every renderer is bit-exact against the reference on every raw
// tensor and every image of every scene, and the images on disk are the ones the verdicts were made on.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Needs no GPU: it reads results/ only, so CI
// runs it on every push (`pnpm fidelity:check`). Regenerating the results needs Chrome and hardware f16
// (`pnpm fidelity:generate`).

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  FIDELITY_CONFIG,
  MIN_TENSORS,
  OTHERS,
  OUTPUTS,
  REFERENCE,
  RENDERERS,
  readJson,
  resultsDir,
  sceneFolders,
} from './suite.mjs';

const problems = [];
const problem = (text) => problems.push(text);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

const configFile = join(resultsDir, 'fidelity.json');
if (!existsSync(configFile)) problem('results/fidelity.json is missing');
else if (JSON.stringify(readJson(configFile)) !== JSON.stringify(FIDELITY_CONFIG)) {
  problem('results/fidelity.json does not match the suite layout in scripts/suite.mjs (regenerate)');
}

const folders = sceneFolders();
if (!folders.length) problem('no scene has an exactness.json');
let tensorCount = 0;
let imageCount = 0;
for (const folder of folders) {
  const dir = join(resultsDir, folder);
  const exactness = readJson(join(dir, 'exactness.json'));
  if (exactness.reference !== REFERENCE) problem(`${folder}: reference is ${exactness.reference}`);
  if (exactness.tensors.length < MIN_TENSORS) {
    problem(`${folder}: ${exactness.tensors.length} tensors compared, expected at least ${MIN_TENSORS}`);
  }
  for (const tensor of exactness.tensors) {
    for (const id of OTHERS) {
      if (tensor[id] !== 'bit-exact')
        problem(`${folder}: ${id} tensor "${tensor.name}": ${JSON.stringify(tensor[id])}`);
    }
  }
  tensorCount += exactness.tensors.length * OTHERS.length;
  for (const { id: output } of OUTPUTS) {
    const row = exactness.images.find((image) => image.output === output);
    if (!row) {
      problem(`${folder}: no verdict for output ${output}`);
      continue;
    }
    for (const renderer of RENDERERS) {
      const file = join(dir, output, `${renderer.id}.png`);
      if (!existsSync(file)) {
        problem(`${folder}: missing ${output}/${renderer.id}.png`);
        continue;
      }
      // Every renderer's file must be the reference's file, byte for byte, and the one the verdict was made on.
      const digest = sha256(readFileSync(file));
      if (digest !== row.sha256) problem(`${folder}: ${output}/${renderer.id}.png differs from the reference's PNG`);
      if (!renderer.reference && row[renderer.id] !== 'bit-exact') {
        problem(`${folder}: ${renderer.id} ${output}: ${JSON.stringify(row[renderer.id])}`);
      }
      imageCount += 1;
    }
  }
  for (const id of OTHERS) {
    if (exactness.summary[id]?.verdict !== 'bit-exact') problem(`${folder}: ${id} summary is not bit-exact`);
  }
  if (!existsSync(join(dir, 'timing.json'))) problem(`${folder}: timing.json is missing`);
}

if (problems.length) {
  for (const text of problems) console.error(`FAIL ${text}`);
  console.error(`${problems.length} problem(s) in ${folders.length} scene(s)`);
  process.exit(1);
}
console.log(
  `parity results OK: ${folders.length} scenes, ${tensorCount} tensor comparisons and ${imageCount} images, ` +
    `${OTHERS.join(' and ')} bit-exact against ${REFERENCE}`,
);
