#!/usr/bin/env node
// Check the reference-wgsl backend (the shim) against the reference WebGPU port running standalone, in real Chrome.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Not part of any default test run: it needs a
// Chrome with WebGPU and `shader-f16` (Windows D3D12 + DXC on the dev machine), and a model directory.
//
// The shim runs the vendored upstream code, so equal bytes here verify what the shim adds: the vendoring, the
// orchestration replacing `Network.create`'s fetches, sharing three's GPUDevice, GPU-resident features/head
// attributes, and the frame adapter (three render target -> the upstream's scene/motion buffers, history). See
// shimPage.ts `parity` for the exact comparisons. Exit code 1 on any differing byte.
//
// Usage (repository root):
//   node packages/three-dlss-nr/test/browser/run-shim-parity-chrome.mjs [--model <dir>] [--size 128x128]
//        [--chrome <path>] [--headed] [--out results.json]

import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  bundlePage,
  evaluateInChrome,
  findChrome,
  option,
  referencePort,
  resolveModelDir,
  servePage,
} from './chrome.mjs';

const args = process.argv.slice(2);
const chrome = option(args, '--chrome') ?? findChrome();
if (!chrome) throw new Error('no Chrome found; pass --chrome <path>');
const [width, height] = (option(args, '--size') ?? '128x128').split('x').map(Number);
const modelDir = await resolveModelDir(args);

const page = await bundlePage(join(import.meta.dirname, 'shimPage.ts'));
const server = await servePage(page, { '/model': modelDir, '/ref': referencePort });
let result;
try {
  const options = JSON.stringify({ modelUrl: `${server.url}model`, shaderBase: `${server.url}ref/`, width, height });
  result = await evaluateInChrome(chrome, server.url, `globalThis.nrShim.parity(${options})`, {
    headed: args.includes('--headed'),
    readyGlobal: 'nrShim',
  });
} finally {
  server.close();
}

console.log(`browser: ${chrome}; adapter: ${result.adapter}; ${result.width}x${result.height}; model: ${modelDir}`);
let failures = 0;
for (const row of result.results) {
  failures += row.mismatches;
  console.log(
    `${row.mismatches ? 'DIFFERS  ' : 'bit-exact'} ${row.name}` +
      (row.bytes ? ` (${row.bytes} bytes)` : '') +
      (row.mismatches ? `: ${row.mismatches} bytes differ, first at ${row.first}` : ''),
  );
}
const out = option(args, '--out');
if (out) writeFileSync(out, JSON.stringify(result, null, 2));
process.exit(failures ? 1 : 0);
