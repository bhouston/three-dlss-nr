#!/usr/bin/env node
// The full-network parity gate: the reference WebGPU port of OpenDLSS-NR (by maan, MIT) and this port's TSL
// `NRNetwork` on one device in real Chrome, on the synthetic weights, byte for byte.
//
// Part of three-dlss-nr. Not part of any default test run: it needs Chrome with WebGPU and `shader-f16` (Windows
// D3D12 + DXC on the dev machine; the reference's FP8 GEMM and window attention need hardware f16, which Dawn in Node
// lacks there and which lavapipe folds). What it checks, per size (networkParityPage.ts `parity`): every captured
// boundary (79), `post merge`, `post merge raw`, `post block raw` and the f32 head identical; both networks give
// identical bytes on a second run; and it prints the reference's boundary statistics (the synthetic-weights
// calibration gate of design 5.1) and our digests. `--golden` rewrites test/network/goldenDigests.ts with them, which
// network.synthetic.gpu.test.ts then checks in Node on any device. Exit code 1 on any difference.
//
// Usage (repository root, after `pnpm build` so the synthetic model can be generated):
//   node packages/three-dlss-nr/test/browser/run-network-parity-chrome.mjs [--sizes 64x64,512x512] [--golden]
//        [--bisect <dispatch count>] [--model <dir>] [--chrome <path>] [--headed] [--out results.json]
// `--bisect N` runs both networks truncated after dispatch N (1..451) and compares every tensor label both allocate:
// the first N where a tensor differs is the first wrong dispatch.

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
} from './chrome.mjs';

const args = process.argv.slice(2);
const chrome = option(args, '--chrome') ?? findChrome();
if (!chrome) throw new Error('no Chrome found; pass --chrome <path>');
const sizes = (option(args, '--sizes') ?? '64x64,512x512').split(',').map((size) => size.split('x').map(Number));
const bisectAt = option(args, '--bisect');
const modelDir = await resolveModelDir(args);

const page = await bundlePage(join(import.meta.dirname, 'networkParityPage.ts'));
const server = await servePage(page, { '/model': modelDir, '/ref': referencePort });
const run = (expression) =>
  evaluateInChrome(chrome, server.url, expression, {
    headed: args.includes('--headed'),
    readyGlobal: 'nrNetworkParity',
    timeoutMs: 3_600_000,
  });

const hex = (value) => (value === undefined ? '' : `0x${value.toString(16).padStart(2, '0')}`);
let failures = 0;
const outputs = [];
const golden = {};
try {
  for (const [width, height] of sizes) {
    const options = { modelUrl: `${server.url}model`, shaderBase: `${server.url}ref/`, width, height };
    if (bisectAt) {
      const result = await run(
        `globalThis.nrNetworkParity.bisect(${JSON.stringify({ ...options, until: Number(bisectAt) })})`,
      );
      outputs.push(result);
      console.log(`${width}x${height}: after dispatch ${result.until} (${result.label}); adapter ${result.adapter}`);
      for (const row of result.results) {
        if (row.mismatches) failures += 1;
        console.log(
          `  ${row.mismatches ? 'DIFFERS  ' : 'equal    '} ${row.name} (${row.bytes} bytes)` +
            (row.mismatches
              ? `: ${row.mismatches} differ, first at ${row.first} ours ${hex(row.ours)} ref ${hex(row.reference)}`
              : ''),
        );
      }
      continue;
    }
    const result = await run(`globalThis.nrNetworkParity.parity(${JSON.stringify(options)})`);
    outputs.push(result);
    console.log(
      `${width}x${height} (field ${result.field}, ${result.dispatches} dispatches); browser ${chrome}; adapter ${result.adapter}`,
    );
    let exact = 0;
    for (const row of result.results) {
      if (row.mismatches) {
        failures += 1;
        console.log(
          `  DIFFERS   ${row.name} (${row.bytes} bytes): ${row.mismatches} differ, first at ${row.first} ` +
            `ours ${hex(row.ours)} ref ${hex(row.reference)}`,
        );
      } else exact += 1;
    }
    console.log(`  bit-exact ${exact} of ${result.results.length} tensors (79 boundaries, 3 post tensors, head)`);
    if (result.repeat.ours || result.repeat.reference) failures += 1;
    console.log(
      `  repeat run: ours ${result.repeat.ours} tensors changed, reference ${result.repeat.reference}; ` +
        `head finite: ${result.finiteHead}`,
    );
    if (!result.finiteHead) failures += 1;
    let outOfRange = 0;
    for (const s of result.stats) {
      const bad = s.saturated >= 0.001 || s.zeros >= 0.3 || s.median < 2 ** -4 || s.median > 8 || s.distinct < 40;
      if (bad) outOfRange += 1;
      if (bad || args.includes('--stats')) {
        console.log(
          `  ${bad ? 'OUT OF RANGE' : 'stats'} ${s.name}: saturated ${(s.saturated * 100).toFixed(3)}%, ` +
            `zeros ${(s.zeros * 100).toFixed(1)}%, median ${s.median}, ${s.distinct} codes`,
        );
      }
    }
    console.log(`  calibration: ${result.stats.length - outOfRange} of ${result.stats.length} boundaries in range`);
    console.log(`  digest ${result.digest.all}`);
    golden[`${width}x${height}`] = result.digest;
  }
} finally {
  server.close();
}

if (args.includes('--golden') && !bisectAt) {
  if (failures) {
    console.error('not writing golden digests: parity failed');
  } else {
    const file = join(packageRoot, 'test/network/goldenDigests.ts');
    writeFileSync(
      file,
      '// Generated by test/browser/run-network-parity-chrome.mjs --golden: digests of the TSL network on the synthetic\n' +
        '// model (seed 1) and syntheticFeatures (seed 1), each tensor verified byte-identical to the reference WebGPU port\n' +
        '// of OpenDLSS-NR (by maan, MIT) in real Chrome on the same device. Do not edit by hand.\n\n' +
        "import type { NetworkDigest } from './golden.js';\n\n" +
        `export const GOLDEN_DIGESTS: Record<string, NetworkDigest> = ${JSON.stringify(golden, null, 2)};\n`,
    );
    console.log(`wrote ${file}`);
  }
}
const out = option(args, '--out');
if (out) writeFileSync(out, JSON.stringify(outputs, null, 2));
process.exit(failures ? 1 : 0);
