// `fetch` for `file:` and `synthetic:` URLs, installed by the `gpu` vitest project (setupFiles).
//
// The reference WebGPU port of OpenDLSS-NR (by maan, MIT; reference/OpenDLSS-NR/ports/browser-webgpu) loads its
// WGSL with `fetch(new URL('shaders/x.wgsl', base))` and its weights with `fetch(`${dir}/manifest.json`)`. Node's
// fetch rejects `file:` URLs, so this wraps `globalThis.fetch`:
//   * `file:`      -> read from disk (GET and HEAD; 404 when missing), so `shaderBase: pathToFileURL(...)` and
//                     `NR_WEIGHTS` / `NR_FIXTURES` directories work;
//   * `synthetic:` -> an in-memory file map registered with `registerSyntheticFiles` (synthetic weights);
//   * anything else -> the original fetch.

import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const synthetic = new Map<string, Uint8Array>();

/**
 * Serve `files` (keys are paths relative to the directory, e.g. `manifest.json`, `model/stages/s00.bin`) under
 * `synthetic://<name>/`. Returns the directory URL to pass to the reference's `Model.load` (no trailing slash).
 */
export function registerSyntheticFiles(name: string, files: ReadonlyMap<string, Uint8Array>): string {
  if (!/^[\w.-]+$/.test(name)) throw new Error(`synthetic directory name "${name}" must be a plain word`);
  const prefix = `synthetic://${name}/`;
  for (const key of synthetic.keys()) if (key.startsWith(prefix)) synthetic.delete(key);
  for (const [path, bytes] of files) synthetic.set(prefix + path.replace(/^\/+/, ''), bytes);
  return `synthetic://${name}`;
}

/** Drop a synthetic directory registered with `registerSyntheticFiles`. */
export function unregisterSyntheticFiles(name: string): void {
  const prefix = `synthetic://${name}/`;
  for (const key of synthetic.keys()) if (key.startsWith(prefix)) synthetic.delete(key);
}

const notFound = (url: string): Response => new Response(`not found: ${url}`, { status: 404, statusText: 'Not Found' });

const bytesResponse = (bytes: Uint8Array, head: boolean): Response =>
  new Response(head ? null : (bytes as unknown as BodyInit), {
    status: 200,
    headers: { 'content-length': String(bytes.byteLength) },
  });

type Fetch = typeof globalThis.fetch;

/** Wrap `globalThis.fetch` (idempotent). */
export function installTestFetch(): void {
  const current = globalThis.fetch as Fetch & { nrTestFetch?: true };
  if (current.nrTestFetch) return;
  const original = current;
  const shim = async (input: Parameters<Fetch>[0], init?: Parameters<Fetch>[1]): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const method = (
      init?.method ?? (typeof input === 'object' && 'method' in input ? input.method : 'GET')
    ).toUpperCase();
    const head = method === 'HEAD';
    if (url.startsWith('file:')) {
      const path = fileURLToPath(url);
      try {
        if (head) {
          const info = await stat(path);
          if (!info.isFile()) return notFound(url);
          return new Response(null, { status: 200, headers: { 'content-length': String(info.size) } });
        }
        return bytesResponse(new Uint8Array(await readFile(path)), false);
      } catch {
        return notFound(url);
      }
    }
    if (url.startsWith('synthetic:')) {
      const bytes = synthetic.get(url);
      return bytes ? bytesResponse(bytes, head) : notFound(url);
    }
    return original(input, init);
  };
  globalThis.fetch = Object.assign(shim, { nrTestFetch: true as const }) as unknown as Fetch;
}

installTestFetch();
