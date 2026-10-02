#!/usr/bin/env node
// Generate the parity suite's results/ in real Chrome: the reference WebGPU port of OpenDLSS-NR (by maan, MIT),
// three-dlss-nr's TSL port and its reference shim on the same features and synthetic weights, per scene and size.
//
// Part of three-dlss-nr. Needs Chrome with WebGPU on a GPU with hardware `shader-f16` (the reference needs it; Dawn
// in Node lacks it on Windows and lavapipe folds its half roundings), and `pnpm build` first (the page imports the
// built library and the synthetic model is generated from it). Uses the repository's DevTools harness
// (packages/three-dlss-nr/test/browser/chrome.mjs); no browser dependency. Run on an otherwise idle GPU: the
// timings are only as good as the machine is quiet.
//
// Usage (repository root):
//   pnpm fidelity:generate [--sizes 256x256,512x512] [--scenes head-front,spheres] [--frames 10] [--warmup 3]
//        [--model <dir>] [--chrome <path>] [--headed]
// Writes packages/fidelity-suite/results/<size>/<scene>/ (PNGs, scene.json, exactness.json, timing.json, README.md),
// results/fidelity.json and results/README.md, then formats the JSON and Markdown with oxfmt.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, sep } from 'node:path';

import {
  bundlePage,
  evaluateInChrome,
  findChrome,
  option,
  referencePort,
  repoRoot,
  resolveModelDir,
} from '../../three-dlss-nr/test/browser/chrome.mjs';
import { encodePng } from './png.mjs';
import {
  FIDELITY_CONFIG,
  OTHERS,
  OUTPUTS,
  REFERENCE,
  RENDERERS,
  resultsDir,
  suiteRoot,
  writeReadmes,
} from './suite.mjs';

const args = process.argv.slice(2);
const chrome = option(args, '--chrome') ?? findChrome();
if (!chrome) throw new Error('no Chrome found; pass --chrome <path>');
const sizes = (option(args, '--sizes') ?? '256x256,512x512').split(',').map((size) => size.split('x').map(Number));
const scenes = option(args, '--scenes')?.split(',');
const frames = Number(option(args, '--frames') ?? 10);
const warmup = Number(option(args, '--warmup') ?? 3);
const modelDir = await resolveModelDir(args);
const modelsDir = join(repoRoot, 'packages/website/public/models');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const SAFE_PATH = /^[\w.-]+(?:\/[\w.-]+)*$/;

/** JSON reports by results-relative path (`<size>/<scene>/parity.json`, `.../timing.json`). */
const reports = new Map();
const written = new Set();

const TYPES = {
  '.json': 'application/json',
  '.wgsl': 'text/plain',
  '.js': 'text/javascript',
  '.glb': 'model/gltf-binary',
};
const mounts = { '/model': modelDir, '/ref': referencePort, '/models': modelsDir };

function collect(request) {
  return new Promise((done, fail) => {
    const parts = [];
    request.on('data', (part) => parts.push(part));
    request.on('end', () => done(Buffer.concat(parts)));
    request.on('error', fail);
  });
}

async function handleUpload(path, url, request) {
  if (!SAFE_PATH.test(path) || path.split('/').includes('..')) throw new Error(`bad upload path ${path}`);
  const body = await collect(request);
  if (path.endsWith('.json')) {
    reports.set(path, JSON.parse(body.toString('utf8')));
    return;
  }
  const width = Number(url.searchParams.get('width'));
  const height = Number(url.searchParams.get('height'));
  const file = join(resultsDir, `${path}.png`);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, encodePng(new Uint8Array(body.buffer, body.byteOffset, body.byteLength), width, height));
  written.add(file);
}

const page = await bundlePage(join(suiteRoot, 'src/page.ts'));
const server = createServer((request, response) => {
  const url = new URL(request.url, 'http://x');
  const path = decodeURIComponent(url.pathname);
  if (request.method === 'POST' && path.startsWith('/upload/')) {
    handleUpload(path.slice('/upload/'.length), url, request).then(
      () => response.writeHead(200).end('ok'),
      (error) => response.writeHead(400).end(String(error)),
    );
    return;
  }
  if (path === '/page.js') return response.writeHead(200, { 'content-type': 'text/javascript' }).end(page);
  if (path === '/' || path === '/index.html') {
    return response
      .writeHead(200, { 'content-type': 'text/html' })
      .end(
        '<!doctype html><meta charset="utf-8"><title>parity suite</title><script type="module" src="/page.js"></script>',
      );
  }
  for (const [prefix, dir] of Object.entries(mounts)) {
    if (!path.startsWith(`${prefix}/`)) continue;
    const file = normalize(join(dir, path.slice(prefix.length + 1)));
    if (!file.startsWith(normalize(dir) + sep) || !existsSync(file) || !statSync(file).isFile()) break;
    response.writeHead(200, {
      'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
      'content-length': statSync(file).size,
    });
    if (request.method === 'HEAD') return response.end();
    return createReadStream(file).pipe(response);
  }
  response.writeHead(404).end(`not found: ${path}`);
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/`;

const sceneIds = scenes ?? ['head-front', 'head-three-quarter', 'spheres', 'checker-room'];
for (const [width, height] of sizes) {
  for (const scene of sceneIds) rmSync(join(resultsDir, `${width}x${height}`, scene), { recursive: true, force: true });
}

const started = performance.now();
let result;
try {
  const options = {
    modelUrl: `${base}model`,
    shaderBase: `${base}ref/`,
    modelsBase: `${base}models/`,
    uploadBase: `${base}upload/`,
    sizes,
    scenes,
    frames,
    warmup,
  };
  result = await evaluateInChrome(chrome, base, `globalThis.nrFidelity.generate(${JSON.stringify(options)})`, {
    headed: args.includes('--headed'),
    readyGlobal: 'nrFidelity',
    timeoutMs: 7_200_000,
  });
} finally {
  server.close();
}
console.log(`browser run done in ${((performance.now() - started) / 1000).toFixed(0)} s on ${result.adapter}`);

/** exactness.json of one scene: tensor verdicts from the page, PNG hashes from disk. */
function exactnessOf(folder, parity) {
  const images = OUTPUTS.map(({ id }) => {
    const hashes = Object.fromEntries(
      RENDERERS.map((r) => [r.id, sha256(readFileSync(join(resultsDir, folder, id, `${r.id}.png`)))]),
    );
    const row = { output: id, sha256: hashes[REFERENCE] };
    for (const other of OTHERS) row[other] = hashes[other] === hashes[REFERENCE] ? 'bit-exact' : 'differs';
    return row;
  });
  const tensors = parity.tensors.map(({ name, bytes, sha256: digest }) => {
    const row = { name, bytes, sha256: digest };
    for (const other of OTHERS) {
      const v = parity.verdicts[other][name];
      row[other] =
        v.verdict === 'bit-exact' ? 'bit-exact' : { verdict: v.verdict, mismatches: v.mismatches, first: v.first };
    }
    return row;
  });
  const summary = Object.fromEntries(
    OTHERS.map((id) => {
      const exactTensors = tensors.filter((t) => t[id] === 'bit-exact').length;
      const exactImages = images.filter((i) => i[id] === 'bit-exact').length;
      const exact = exactTensors === tensors.length && exactImages === images.length;
      return [
        id,
        {
          verdict: exact ? 'bit-exact' : 'differs',
          tensors: `${exactTensors}/${tensors.length}`,
          images: `${exactImages}/${images.length}`,
        },
      ];
    }),
  );
  return {
    title: parity.title,
    description: parity.description,
    tags: parity.tags,
    size: parity.size,
    adapter: parity.adapter,
    reference: REFERENCE,
    geometry: parity.geometry,
    dispatches: parity.dispatches,
    blendScale: parity.blendScale,
    summary,
    tensors,
    images,
  };
}

const toFormat = [];
const writeText = (file, text) => {
  writeFileSync(file, text);
  toFormat.push(file);
};
for (const [width, height] of sizes) {
  for (const scene of sceneIds) {
    const folder = `${width}x${height}/${scene}`;
    const parity = reports.get(`${folder}/parity.json`);
    const timing = reports.get(`${folder}/timing.json`);
    if (!parity || !timing) throw new Error(`${folder}: the page sent no ${parity ? 'timing' : 'parity'} report`);
    const exactness = exactnessOf(folder, parity);
    const dir = join(resultsDir, folder);
    writeText(join(dir, 'scene.json'), JSON.stringify({ title: exactness.title, tags: exactness.tags }, null, 2));
    writeText(join(dir, 'exactness.json'), JSON.stringify(exactness, null, 2));
    writeText(join(dir, 'timing.json'), JSON.stringify(timing, null, 2));
    console.log(
      `${folder}: ${OTHERS.map((id) => `${id} ${exactness.summary[id].verdict} (tensors ${exactness.summary[id].tensors}, images ${exactness.summary[id].images})`).join('; ')}; ` +
        `GPU median ms ${RENDERERS.map((r) => `${r.id} ${timing.renderers[r.id].gpu?.median.toFixed(1)}`).join(', ')}`,
    );
  }
}
writeText(join(resultsDir, 'fidelity.json'), JSON.stringify(FIDELITY_CONFIG, null, 2));
toFormat.push(...writeReadmes());
execFileSync(process.execPath, [join(repoRoot, 'node_modules/oxfmt/bin/oxfmt'), ...toFormat], {
  cwd: repoRoot,
  stdio: 'inherit',
});
console.log(`wrote ${written.size} images under ${resultsDir}; check with: pnpm fidelity:check`);
