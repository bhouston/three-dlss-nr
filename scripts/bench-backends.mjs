#!/usr/bin/env node
// Benchmark the NR backends (native TSL port vs the reference-wgsl shim) in real Chrome.
//
// three-dlss-nr is a port to three.js of OpenDLSS-NR by maan (MIT, https://github.com/maanHimself/OpenDLSS-NR); the
// reference-wgsl backend runs that project's own WebGPU port unchanged on three's GPUDevice. Both backends are
// created on the same renderer/device, fed the same features, and timed the same way (NRFrameTimer: timestamp
// queries bracketing the frame on the queue when `timestamp-query` is available, else submit -> onSubmittedWorkDone).
// Chrome is used because Dawn in Node has no `shader-f16` on the Windows dev machine.
//
// Usage (repository root, after `pnpm build`):
//   node scripts/bench-backends.mjs [--model <dir>] [--sizes 512x512,1280x720] [--frames 30] [--warmup 5]
//        [--backends tsl,reference-wgsl] [--chrome <path>] [--headed] [--out bench.json]
// --model: a model directory (manifest.json + model/stages/*); defaults to $NR_MODEL_DIR or the synthetic model
// (generated once by the package's synthetic generator). Only synthetic or user-supplied weights; never NVIDIA's.
// The TSL backend is benchmarked once the package index exports `tslBackend` (design chunk E); until then the shim runs
// alone.
//
// Run on an otherwise idle GPU: anything else using the GPU (games, video, another benchmark or test run) skews the
// numbers, and nothing here can detect it.

import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  bundlePage,
  evaluateInChrome,
  findChrome,
  option,
  packageRoot,
  referencePort,
  resolveModelDir,
  servePage,
} from '../packages/three-dlss-nr/test/browser/chrome.mjs';

const args = process.argv.slice(2);
const chrome = option(args, '--chrome') ?? findChrome();
if (!chrome) throw new Error('no Chrome found; pass --chrome <path>');
const sizes = (option(args, '--sizes') ?? '512x512,1280x720').split(',').map((size) => size.split('x').map(Number));
const frames = Number(option(args, '--frames') ?? 30);
const warmup = Number(option(args, '--warmup') ?? 5);
const only = option(args, '--backends')?.split(',');
const modelDir = await resolveModelDir(args);

console.warn(
  'WARNING: run this on an idle GPU. Other GPU work (another benchmark, tests, video, games) makes the numbers ' +
    'meaningless.',
);
// Best effort: NVIDIA's tool reports utilization; other vendors are not checked.
const busy = (() => {
  try {
    const text = execFileSync('nvidia-smi', ['--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return Math.max(...text.split('\n').filter(Boolean).map(Number));
  } catch {
    return null;
  }
})();
if (busy !== null && busy > 10) {
  console.warn(`WARNING: the GPU is ${busy}% busy before the benchmark starts: these numbers are UNRELIABLE.`);
}

const page = await bundlePage(join(packageRoot, 'test/browser/shimPage.ts'));
const server = await servePage(page, { '/model': modelDir, '/ref': referencePort });
let result;
try {
  const options = JSON.stringify({ modelUrl: `${server.url}model`, sizes, frames, warmup, only });
  result = await evaluateInChrome(chrome, server.url, `globalThis.nrShim.bench(${options})`, {
    headed: args.includes('--headed'),
    readyGlobal: 'nrShim',
  });
} finally {
  server.close();
}

const fmt = (summary) => (summary ? `${summary.min.toFixed(2)} / ${summary.median.toFixed(2)}` : 'n/a');
console.log(`\nbrowser: ${chrome}`);
console.log(`adapter: ${result.adapter}`);
console.log(`model:   ${modelDir}; ${frames} timed frames after ${warmup} warm-up frames\n`);
console.log(
  '| backend | size (field) | GPU ms min / median | wall ms min / median | create ms | first frame ms | activations MiB | weights MiB |',
);
console.log('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |');
for (const row of result.rows) {
  if (row.skipped) {
    console.log(`| ${row.backend} | skipped: ${row.skipped} | | | | | | |`);
    continue;
  }
  console.log(
    `| ${row.backend} | ${row.size} (${row.field}) | ${fmt(row.gpu)} | ${fmt(row.wall)} | ` +
      `${row.createMilliseconds.toFixed(0)} | ${row.firstFrameMilliseconds.toFixed(0)} | ` +
      `${row.activationMiB.toFixed(0)} | ${row.weightMiB.toFixed(0)} |` +
      (row.headFinite ? '' : ' (head not finite!)'),
  );
}
console.log(
  '\nGPU ms: timestamp queries around the frame (method: ' +
    `${result.rows.find((row) => row.method)?.method ?? 'n/a'}); wall ms: submit to onSubmittedWorkDone. ` +
    (busy !== null && busy > 10
      ? `UNRELIABLE: the GPU was ${busy}% busy at the start.`
      : 'Numbers are only meaningful on an idle GPU.'),
);
const out = option(args, '--out');
if (out) writeFileSync(out, JSON.stringify({ chrome, modelDir, frames, warmup, ...result }, null, 2));
