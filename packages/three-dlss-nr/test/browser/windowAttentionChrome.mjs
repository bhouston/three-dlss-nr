#!/usr/bin/env node
// Run the reference's composed window attention in a real Chrome and compare it with our oracle, byte for byte.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT, pinned at 9d08f41). The reference's window
// attention needs `shader-f16`, which Dawn in Node lacks on the Windows dev machine (design 0.2), and lavapipe's half
// arithmetic is suspect; Chrome on D3D12 with DXC has it. This one-off harness:
//   1. bundles test/browser/windowAttentionCases.ts with esbuild and builds the cases and the oracle's answers
//      (the same inputs as src/kernels/windowAttention.gpu.test.ts);
//   2. serves the reference port (reference/OpenDLSS-NR/ports/browser-webgpu) and a blank page over HTTP;
//   3. launches Chrome (headless, DevTools protocol, no dependency), which imports the reference's own
//      src/window/variants.js and shaders/numerics.wgsl and dispatches `attend_window_tiled` exactly as graph.js
//      `windowAttention` does (module `enable f16;` + numerics + windowAttentionCode(), the six overrides, bindings
//      1 qkv / 2 scales / 3 prior / 6 output, [heads, min(tasks, 65535), ceil(tasks / 65535)]);
//   4. compares the reference's bytes (whole allocation, sentinel 0xCD past the valid rows) with the oracle's.
// Since windowAttention.gpu.test.ts proves our TSL equals the oracle on these inputs, a pass ties our TSL to the
// reference. Usage (from the repository root):
//   node packages/three-dlss-nr/test/browser/windowAttentionChrome.mjs [--chrome <path>] [--headed] [--out <json>]

import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'esbuild';

const here = dirname(fileURLToPath(import.meta.url));
const repository = resolve(here, '../../../..');
const referencePort = join(repository, 'reference/OpenDLSS-NR/ports/browser-webgpu');

const argument = (flag, fallback) => {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : fallback;
};
const chromePath = argument('--chrome', 'C:/Program Files/Google/Chrome/Application/chrome.exe');
const headed = process.argv.includes('--headed');
const outPath = argument('--out', undefined);

/** Bundle the cases module (TypeScript, imports the oracle) and build the cases in Node. */
async function loadCases() {
  const scratch = mkdtempSync(join(tmpdir(), 'nr-window-cases-'));
  const bundle = join(scratch, 'cases.mjs');
  await build({
    entryPoints: [join(here, 'windowAttentionCases.ts')],
    bundle: true,
    platform: 'node',
    format: 'esm',
    outfile: bundle,
    logLevel: 'warning',
  });
  const { windowCases } = await import(pathToFileURL(bundle).href);
  const cases = windowCases();
  rmSync(scratch, { recursive: true, force: true });
  return cases;
}

const PAGE = `<!doctype html><meta charset="utf-8"><title>window attention reference</title>
<script type="module">
import { windowAttentionCode, WINDOW_QUERIES } from '/ref/src/window/variants.js';
const bytesOf = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
const textOf = (bytes) => { let s = ''; for (const b of bytes) s += String.fromCharCode(b); return btoa(s); };
globalThis.runWindowCases = async (cases) => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter.features.has('shader-f16')) throw new Error('this adapter has no shader-f16');
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 512,
                      maxComputeWorkgroupSizeX: 512 },
  });
  const numerics = await (await fetch('/ref/shaders/numerics.wgsl')).text();
  const module = device.createShaderModule({ code: ['enable f16;', numerics, windowAttentionCode()].join('\\n') });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) if (m.type === 'error') throw new Error('window module: ' + m.message);
  const storage = (bytes, usage) => {
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(bytes.byteLength / 4) * 4 + 4), usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(bytes);
    buffer.unmap();
    return buffer;
  };
  const results = [];
  for (const c of cases) {
    const tokens = c.width * c.height;
    const channels = c.heads * 32;
    const allocRows = Math.ceil(tokens / 64) * 64;
    const outputBytes = allocRows * channels;
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'attend_window_tiled', constants: {
        WINDOW_WIDTH: c.width, WINDOW_HEIGHT: c.height, WINDOW_CHANNELS: channels,
        WINDOW_SHIFT_X: c.shiftX, WINDOW_SHIFT_Y: c.shiftY, WINDOW_USE_RELATIVE_BIAS: 1 } },
    });
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const qkvBytes = new Uint8Array(allocRows * channels * 3 * 2);
    qkvBytes.set(bytesOf(c.qkv));
    const output = storage(new Uint8Array(outputBytes).fill(0xcd), S);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 1, resource: { buffer: storage(qkvBytes, S) } },
      { binding: 2, resource: { buffer: storage(bytesOf(c.scales), S) } },
      { binding: 3, resource: { buffer: storage(bytesOf(c.prior), S) } },
      { binding: 6, resource: { buffer: output } },
    ] });
    const tasks = Math.ceil((c.width + c.shiftX) / 8) * Math.ceil((c.height + c.shiftY) / 8) * (64 / WINDOW_QUERIES);
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(c.heads, Math.min(tasks, 65535), Math.ceil(tasks / 65535));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, output.size);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error) throw new Error(c.name + ': ' + error.message);
    await read.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(read.getMappedRange().slice(0, outputBytes));
    read.unmap();
    results.push({ name: c.name, bytes: textOf(bytes) });
  }
  return { adapter: adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description].join(' ') : '', results };
};
globalThis.harnessReady = true;
</script>`;

const MIME = { '.js': 'text/javascript', '.wgsl': 'text/plain', '.html': 'text/html' };

function serve() {
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    if (path === '/' || path === '/index.html') {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.end(PAGE);
      return;
    }
    if (path.startsWith('/ref/')) {
      const file = resolve(referencePort, path.slice(5));
      if (file.startsWith(referencePort)) {
        try {
          const body = readFileSync(file);
          response.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
          response.end(body);
          return;
        } catch {
          // fall through to 404
        }
      }
    }
    response.writeHead(404);
    response.end();
  });
  return new Promise((resolveServer) => server.listen(0, '127.0.0.1', () => resolveServer(server)));
}

/** Evaluate `expression` (a promise) in a page of a fresh Chrome over the DevTools protocol. */
async function evaluateInChrome(url, expression) {
  const profile = mkdtempSync(join(tmpdir(), 'nr-chrome-'));
  const chrome = spawn(
    chromePath,
    [
      ...(headed ? [] : ['--headless=new']),
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--enable-unsafe-webgpu',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  try {
    const endpoint = await new Promise((resolveEndpoint, reject) => {
      let text = '';
      chrome.stderr.on('data', (chunk) => {
        text += chunk;
        const match = /DevTools listening on (ws:\/\/\S+)/.exec(text);
        if (match) resolveEndpoint(match[1]);
      });
      chrome.on('exit', (code) => reject(new Error(`chrome exited (${code}): ${text}`)));
    });
    const port = new URL(endpoint).port;
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${url}`, { method: 'PUT' })).json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolveOpen, reject) => {
      socket.addEventListener('open', resolveOpen);
      socket.addEventListener('error', reject);
    });
    let nextId = 0;
    const pending = new Map();
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      const waiter = pending.get(message.id);
      if (waiter) {
        pending.delete(message.id);
        waiter(message);
      }
    });
    const send = (method, params = {}) =>
      new Promise((resolveReply) => {
        const id = ++nextId;
        pending.set(id, resolveReply);
        socket.send(JSON.stringify({ id, method, params }));
      });
    await send('Runtime.enable');
    for (let attempt = 0; attempt < 200; ++attempt) {
      const probe = await send('Runtime.evaluate', {
        expression: 'globalThis.harnessReady === true',
        returnByValue: true,
      });
      if (probe.result?.result?.value === true) break;
      await new Promise((wake) => setTimeout(wake, 100));
    }
    const reply = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    socket.close();
    if (reply.result?.exceptionDetails) {
      throw new Error(
        `page: ${reply.result.exceptionDetails.exception?.description ?? reply.result.exceptionDetails.text}`,
      );
    }
    return reply.result.result.value;
  } finally {
    chrome.kill();
    await new Promise((wake) => setTimeout(wake, 500));
    rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
}

const cases = await loadCases();
const server = await serve();
const url = `http://127.0.0.1:${server.address().port}/`;
let failures = 0;
try {
  const payload = cases.map(({ expected: _expected, ...input }) => input);
  const { adapter, results } = await evaluateInChrome(url, `runWindowCases(${JSON.stringify(payload)})`);
  console.log(`Chrome adapter: ${adapter}`);
  for (const [index, testCase] of cases.entries()) {
    const reference = Buffer.from(results[index].bytes, 'base64');
    const expected = Buffer.from(testCase.expected, 'base64');
    let mismatches = 0;
    let first = '';
    for (let i = 0; i < reference.length; ++i) {
      const want = i < expected.length ? expected[i] : 0xcd;
      if (reference[i] !== want) {
        if (!mismatches)
          first = ` first at byte ${i}: reference 0x${reference[i].toString(16)} oracle 0x${want.toString(16)}`;
        mismatches += 1;
      }
    }
    if (mismatches) failures += 1;
    console.log(
      `${mismatches ? 'FAIL' : 'ok  '} ${testCase.name}: ${reference.length} bytes, ${mismatches} differ${first}`,
    );
  }
  if (outPath) writeFileSync(outPath, JSON.stringify({ adapter, results }, null, 1));
} finally {
  server.close();
}
console.log(failures ? `${failures} of ${cases.length} cases differ` : `all ${cases.length} cases bit-exact`);
process.exitCode = failures ? 1 : 0;
