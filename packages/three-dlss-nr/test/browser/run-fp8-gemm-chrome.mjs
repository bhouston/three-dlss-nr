#!/usr/bin/env node
// One-off check: the reference's composed FP8 GEMM on real f16 hardware (a real Chrome), against our CPU oracle.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Not part of any test run. Why it exists:
// Dawn in Node has no `shader-f16` on the Windows dev machine, and lavapipe (CI) does but evaluates half arithmetic
// in a way that disagrees with both NVIDIA hardware and the oracle (see three-dlss-nr-design.md, "[B: lavapipe]").
// Chrome on Windows (D3D12 + DXC) exposes `shader-f16` on the RTX 3060 Ti, so the reference runs there as it does
// for its users.
//
// What it does: bundles fp8GemmPage.ts with esbuild, serves it on 127.0.0.1, opens it in Chrome (DevTools protocol, no dependency),
// runs the seven FP8 cases of refKernels.gpu.test.ts (same inputs, byte for byte; fp8Cases.ts), and compares every
// published byte / half with oracleGemmFp8, and the reference's GPU-built SiLU table with ours (tables.ts). Exit code 1
// on any difference.
//
// Usage (from the repository root):
//   node packages/three-dlss-nr/test/browser/run-fp8-gemm-chrome.mjs [--chrome <path>] [--headed] [--out results.json]
// Default browser: Chrome stable at its Windows / Linux install path.

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'esbuild';

// Chrome over the DevTools protocol, no dependency: launch with a debugging port, open the page, await an expression.
async function evaluateInChrome(executable, url, expression, { headed = false } = {}) {
  const profile = mkdtempSync(join(tmpdir(), 'nr-chrome-'));
  const chrome = spawn(
    executable,
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
    const endpoint = await new Promise((resolve, reject) => {
      let text = '';
      chrome.stderr.on('data', (chunk) => {
        text += chunk;
        const match = /DevTools listening on (ws:\/\/\S+)/.exec(text);
        if (match) resolve(match[1]);
      });
      chrome.on('exit', (code) => reject(new Error(`chrome exited (${code}): ${text}`)));
    });
    const port = new URL(endpoint).port;
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${url}`, { method: 'PUT' })).json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener('error', reject, { once: true });
    });
    let nextId = 0;
    const pending = new Map();
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.method === 'Runtime.consoleAPICalled') {
        console.log(`[page] ${message.params.args.map((a) => a.value ?? a.description).join(' ')}`);
      }
      const waiter = pending.get(message.id);
      if (waiter) {
        pending.delete(message.id);
        waiter(message);
      }
    });
    const send = (method, params = {}) =>
      new Promise((resolve) => {
        const id = ++nextId;
        pending.set(id, resolve);
        socket.send(JSON.stringify({ id, method, params }));
      });
    await send('Runtime.enable');
    // Wait for the page script to define the entry point.
    for (let i = 0; i < 100; ++i) {
      const probe = await send('Runtime.evaluate', { expression: 'typeof globalThis.runFp8Gemm', returnByValue: true });
      if (probe.result?.result?.value === 'function') break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const reply = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    socket.close();
    if (reply.result?.exceptionDetails) {
      const details = reply.result.exceptionDetails;
      throw new Error(`page threw: ${details.exception?.description ?? details.text}`);
    }
    return reply.result.result.value;
  } finally {
    chrome.kill();
  }
}

const here = fileURLToPath(new URL('.', import.meta.url));
const repo = join(here, '../../../..');
const alias = { '@ref': join(repo, 'reference/OpenDLSS-NR/ports/browser-webgpu/src') };

const args = process.argv.slice(2);
const option = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const chromePath =
  option('--chrome') ??
  ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find((p) => existsSync(p));

// The page bundle (browser) and the oracle (Node).
const page = await build({
  entryPoints: [join(here, 'fp8GemmPage.ts')],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  write: false,
  alias,
  logLevel: 'warning',
});
const outDir = mkdtempSync(join(tmpdir(), 'nr-fp8-'));
await build({
  entryPoints: [join(here, 'fp8GemmOracle.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: join(outDir, 'oracle.mjs'),
  alias,
  logLevel: 'warning',
});
const { fp8OracleResults, fp8OracleSiluCodes } = await import(pathToFileURL(join(outDir, 'oracle.mjs')).href);

const html = '<!doctype html><meta charset="utf-8"><title>fp8 gemm</title><script src="page.js"></script>';
const server = createServer((request, response) => {
  if (request.url === '/page.js') {
    response.writeHead(200, { 'content-type': 'text/javascript' });
    response.end(page.outputFiles[0].contents);
  } else {
    response.writeHead(200, { 'content-type': 'text/html' });
    response.end(html);
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;

if (!chromePath) throw new Error('no Chrome found; pass --chrome <path>');
let result;
try {
  result = await evaluateInChrome(chromePath, url, 'globalThis.runFp8Gemm()', { headed: args.includes('--headed') });
} finally {
  server.close();
}

console.log(`browser: ${chromePath}; adapter: ${result.adapter}`);
const expected = fp8OracleResults();
const hex = (v, digits) => `0x${v.toString(16).padStart(digits, '0')}`;
let failures = 0;
for (const [i, ours] of result.cases.entries()) {
  const want = expected[i];
  for (const [kind, digits] of [
    ['e4', 2],
    ['half', 4],
  ]) {
    if (!ours[kind]) continue;
    const diffs = [];
    ours[kind].forEach((value, j) => {
      if (value !== want[kind][j]) diffs.push(`[${j}] hw ${hex(value, digits)} oracle ${hex(want[kind][j], digits)}`);
    });
    failures += diffs.length;
    console.log(
      `${diffs.length === 0 ? 'bit-exact' : 'DIFFERS  '} ${ours.label} ${kind}: ${diffs.length} of ${ours[kind].length}` +
        (diffs.length ? `: ${diffs.slice(0, 8).join(', ')}` : ''),
    );
  }
}
// The SiLU table, for every half input (NaN inputs excluded: WGSL leaves min/max of NaN unspecified).
{
  const want = fp8OracleSiluCodes();
  const diffs = [];
  result.siluCodes.forEach((code, h) => {
    const nan = (h & 0x7c00) === 0x7c00 && (h & 0x3ff) !== 0;
    if (!nan && code !== want[h]) diffs.push(`[${hex(h, 4)}] hw ${hex(code, 2)} oracle ${hex(want[h], 2)}`);
  });
  failures += diffs.length;
  console.log(
    `${diffs.length === 0 ? 'bit-exact' : 'DIFFERS  '} SiLU table (packedSiluTable): ${diffs.length} of 63490 non-NaN halves` +
      (diffs.length ? `: ${diffs.slice(0, 8).join(', ')}` : ''),
  );
}
const out = option('--out');
if (out) writeFileSync(out, JSON.stringify({ ...result, oracle: expected }));
process.exit(failures ? 1 : 0);
