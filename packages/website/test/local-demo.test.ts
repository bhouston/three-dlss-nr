// Execute the shipped local launcher; replace only its DOM, local-file HTTP and GPU controller boundaries.
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DemoState } from '../src/lib/demo';

const html = readFileSync(new URL('../public/local-demo.html', import.meta.url), 'utf8');
const script = html.match(/<script type="module">([\s\S]*?)<\/script>/)![1];

function launch(loadState: 'building' | 'ready' = 'building') {
  // The launcher consumes these state fields. GPU compilation is deliberately deferred past loadDirectory().
  const state: Pick<DemoState, 'phase' | 'error' | 'weightsError' | 'modelError' | 'network' | 'backends'> = {
    phase: 'ready',
    error: null,
    weightsError: null,
    modelError: null,
    network: { state: 'none', message: 'NR off: no weights loaded', since: 0 },
    backends: [{ id: 'reference-wgsl', label: 'Reference WGSL', reason: null, detail: null }],
  };
  const listeners = new Set<() => void>();
  const status = {
    textContent: 'A helyi modell betöltése…',
    style: { background: '#182a53' },
    removed: false,
    remove() {
      this.removed = true;
    },
  };
  const controller = {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setResolution: async () => {},
    setSettings() {},
    setModel: async () => {},
    setAutoRotate() {},
    setBackend: async () => {},
    setView() {},
    async loadDirectory() {
      state.network = { state: loadState, message: 'compiling kernels 16/420', since: 0 };
    },
  };
  const frame = { contentDocument: { readyState: 'complete' }, contentWindow: { __nrDemo: controller } };
  const window: { __localModelReady?: boolean } = {};
  runInNewContext(script, {
    document: { querySelector: (selector: string) => (selector === '#demo' ? frame : status) },
    window,
    location: { search: '' },
    URLSearchParams,
    TextDecoder,
    File,
    // No model weights are downloaded or generated: the fake GPU boundary needs only a manifest envelope.
    fetch: async () => new Response(JSON.stringify({ stages: [] })),
    setTimeout,
    Date,
    console: { error() {} },
  });
  const update = (patch: Partial<typeof state>) => {
    Object.assign(state, patch);
    for (const listener of listeners) listener();
  };
  return { status, window, update, listeners };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('local model launcher', () => {
  it('waits through kernel compilation after loadDirectory resolves, then removes the banner', async () => {
    const app = launch();
    await vi.advanceTimersByTimeAsync(0);
    expect(app.status.style.background).not.toBe('#8a1d24');
    expect(app.window.__localModelReady).not.toBe(true);

    // Slow first-time compilation is still progress, even beyond the launcher's initial WebGPU timeout.
    await vi.advanceTimersByTimeAsync(180_000);
    expect(app.status.style.background).not.toBe('#8a1d24');
    expect(app.status.removed).toBe(false);

    app.update({ network: { state: 'building', message: 'compiling kernels 48/420', since: 0 } });
    await vi.advanceTimersByTimeAsync(100);
    expect(app.status.textContent).toContain('48/420');

    app.update({ network: { state: 'ready', message: 'ready', since: 0 } });
    await vi.advanceTimersByTimeAsync(100);
    expect(app.window.__localModelReady).toBe(true);
    expect(app.status.removed).toBe(true);
    expect(app.listeners.size).toBe(0);
  });

  it('handles an already ready network without waiting for another state event', async () => {
    const app = launch('ready');
    await vi.advanceTimersByTimeAsync(0);
    expect(app.window.__localModelReady).toBe(true);
    expect(app.status.removed).toBe(true);
    expect(app.listeners.size).toBe(0);
  });

  it.each([
    [
      'network failure',
      { network: { state: 'failed', message: 'GPU validation failed', since: 0 } },
      'GPU validation failed',
    ],
    ['weights error', { weightsError: 'Missing stage file' }, 'Missing stage file'],
    ['device loss', { phase: 'error', error: 'The GPU stopped responding' }, 'The GPU stopped responding'],
  ] as const)('reports a real %s that occurs while waiting', async (_name, patch, message) => {
    const app = launch();
    await vi.advanceTimersByTimeAsync(0);
    app.update(patch);
    await vi.advanceTimersByTimeAsync(100);
    expect(app.status.style.background).toBe('#8a1d24');
    expect(app.status.textContent).toContain(message);
    expect(app.window.__localModelReady).not.toBe(true);
    expect(app.status.removed).toBe(false);
    expect(app.listeners.size).toBe(0);
  });
});
