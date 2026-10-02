// A minimal real-browser harness: bundle a page with esbuild, serve it (plus static directories) on 127.0.0.1, open
// it in Chrome over the DevTools protocol (no dependency), and await an expression.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Why a real browser: Dawn in Node exposes no
// `shader-f16` on the Windows dev machine (and lavapipe's half arithmetic is not trustworthy), while Chrome on
// D3D12 + DXC does, so the reference WebGPU port - and the reference-wgsl backend - run there as for their users.
// Used by test/browser/run-shim-parity-chrome.mjs and scripts/bench-backends.mjs; never part of `pnpm test`.

import { spawn } from 'node:child_process';
import { createReadStream, existsSync, mkdtempSync, readdirSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'esbuild';

const here = fileURLToPath(new URL('.', import.meta.url));
export const packageRoot = resolve(here, '../..');
export const repoRoot = resolve(packageRoot, '../..');
export const referencePort = join(repoRoot, 'reference/OpenDLSS-NR/ports/browser-webgpu');

/** Chrome stable, else Playwright's Chromium (%LOCALAPPDATA%\ms-playwright\chromium-*), else null. */
export function findChrome() {
  const candidates = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/usr/bin/google-chrome',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ];
  const playwright = process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, 'ms-playwright') : null;
  if (playwright && existsSync(playwright)) {
    for (const entry of readdirSync(playwright)
      .filter((name) => /^chromium-\d+$/.test(name))
      .toSorted()
      .toReversed()) {
      candidates.push(
        join(playwright, entry, 'chrome-win64', 'chrome.exe'),
        join(playwright, entry, 'chrome-win', 'chrome.exe'),
      );
    }
  }
  return candidates.find((path) => existsSync(path)) ?? null;
}

/** Bundle `entry` (TypeScript) for the browser as one ES module. */
export async function bundlePage(entry) {
  const result = await build({
    entryPoints: [entry],
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    write: false,
    alias: { '@ref': join(referencePort, 'src') },
    logLevel: 'warning',
  });
  return result.outputFiles[0].contents;
}

const TYPES = {
  '.json': 'application/json',
  '.wgsl': 'text/plain',
  '.js': 'text/javascript',
  '.bin': 'application/octet-stream',
};

/**
 * Serve the page (`/` and `/page.js`) and static directories (`mounts`: { '/model': dir }) on an ephemeral port.
 * Returns { url, close }.
 */
export async function servePage(pageJs, mounts = {}) {
  const html =
    '<!doctype html><meta charset="utf-8"><title>three-dlss-nr browser harness</title>' +
    '<script type="module" src="/page.js"></script>';
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    if (path === '/page.js') {
      response.writeHead(200, { 'content-type': 'text/javascript' });
      response.end(pageJs);
      return;
    }
    for (const [prefix, dir] of Object.entries(mounts)) {
      if (!path.startsWith(`${prefix}/`)) continue;
      const file = normalize(join(dir, path.slice(prefix.length + 1)));
      if (!file.startsWith(normalize(dir) + sep) || !existsSync(file) || !statSync(file).isFile()) break;
      response.writeHead(200, {
        'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
        'content-length': statSync(file).size,
      });
      if (request.method === 'HEAD') response.end();
      else createReadStream(file).pipe(response);
      return;
    }
    if (path === '/' || path === '/index.html') {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.end(html);
      return;
    }
    response.writeHead(404);
    response.end(`not found: ${path}`);
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  return { url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() };
}

/**
 * Open `url` in Chrome, wait until `globalThis[readyGlobal]` exists, then await `expression` (returnByValue).
 * Page console output is echoed with a `[page]` prefix.
 */
export async function evaluateInChrome(
  executable,
  url,
  expression,
  { headed = false, readyGlobal, timeoutMs = 600_000 } = {},
) {
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
      // Full-resolution timestamp queries (Chrome quantizes them to 100 us otherwise).
      '--enable-webgpu-developer-features',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  let socket;
  try {
    const endpoint = await new Promise((done, fail) => {
      let text = '';
      chrome.stderr.on('data', (chunk) => {
        text += chunk;
        const match = /DevTools listening on (ws:\/\/\S+)/.exec(text);
        if (match) done(match[1]);
      });
      chrome.on('exit', (code) => fail(new Error(`chrome exited (${code}): ${text}`)));
    });
    const port = new URL(endpoint).port;
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${url}`, { method: 'PUT' })).json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((done, fail) => {
      socket.addEventListener('open', done, { once: true });
      socket.addEventListener('error', fail, { once: true });
    });
    let nextId = 0;
    const pending = new Map();
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.method === 'Runtime.consoleAPICalled') {
        console.log(`[page] ${message.params.args.map((a) => a.value ?? a.description).join(' ')}`);
      }
      if (message.method === 'Runtime.exceptionThrown') {
        console.error(`[page] uncaught: ${message.params.exceptionDetails?.exception?.description}`);
      }
      const waiter = pending.get(message.id);
      if (waiter) {
        pending.delete(message.id);
        waiter(message);
      }
    });
    const send = (method, params = {}) =>
      new Promise((done) => {
        const id = ++nextId;
        pending.set(id, done);
        socket.send(JSON.stringify({ id, method, params }));
      });
    await send('Runtime.enable');
    if (readyGlobal) {
      let ready = false;
      for (let i = 0; i < 300 && !ready; ++i) {
        const probe = await send('Runtime.evaluate', {
          expression: `typeof globalThis.${readyGlobal}`,
          returnByValue: true,
        });
        ready = probe.result?.result?.value === 'object' || probe.result?.result?.value === 'function';
        if (!ready) await new Promise((done) => setTimeout(done, 100));
      }
      if (!ready) throw new Error(`the page never defined ${readyGlobal}`);
    }
    const reply = await Promise.race([
      send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
      new Promise((_, fail) => setTimeout(() => fail(new Error(`timed out after ${timeoutMs} ms`)), timeoutMs)),
    ]);
    if (reply.result?.exceptionDetails) {
      const details = reply.result.exceptionDetails;
      throw new Error(`page threw: ${details.exception?.description ?? details.text}`);
    }
    return reply.result.result.value;
  } finally {
    socket?.close();
    chrome.kill();
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      // Chrome may still hold the profile for a moment on Windows; it is a temp directory.
    }
  }
}

/** `--name value` from argv. */
export const option = (args, name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};

/**
 * The model directory to serve: `--model <dir>`, else `$NR_MODEL_DIR`, else a synthetic model generated once into
 * node_modules/.cache/three-dlss-nr/synthetic-model by the package's synthetic generator (design chunk A,
 * dist/synthetic/generate.js after `pnpm build`). Synthetic weights only; never NVIDIA's.
 */
export async function resolveModelDir(args) {
  const given = option(args, '--model') ?? process.env.NR_MODEL_DIR;
  if (given) {
    if (!existsSync(join(given, 'manifest.json'))) throw new Error(`${given} has no manifest.json`);
    return resolve(given);
  }
  const cache = join(repoRoot, 'node_modules/.cache/three-dlss-nr/synthetic-model');
  if (existsSync(join(cache, 'manifest.json'))) return cache;
  const generator = join(packageRoot, 'dist/synthetic/generate.js');
  if (!existsSync(generator)) {
    // TODO(chunk A): once src/synthetic/generate.ts is merged this branch is taken automatically after `pnpm build`.
    throw new Error(
      'no model directory: pass --model <dir> (a model directory: manifest.json + model/stages/*, e.g. from ' +
        '`node scripts/make-synthetic-model.mjs <dir>`), or build the synthetic generator first',
    );
  }
  const { generateSyntheticModel } = await import(pathToFileURL(generator).href);
  const { files } = await generateSyntheticModel({ seed: 1 });
  const { mkdir, writeFile } = await import('node:fs/promises');
  const { dirname } = await import('node:path');
  for (const [path, data] of files) {
    const file = join(cache, ...path.split('/'));
    await mkdir(dirname(file), { recursive: true });
    await writeFile(file, data);
  }
  return cache;
}
