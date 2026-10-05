import { afterEach, describe, expect, it, vi } from 'vitest';
import * as nr from 'three-dlss-nr';

import { DemoController } from './demo';
import type { WeightsSource } from './weights';

const weights = vi.hoisted(() => ({ read: vi.fn(), synthetic: vi.fn() }));
vi.mock('./weights', () => ({ readModelDirectory: weights.read, syntheticWeights: weights.synthetic }));

function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const source = (label = 'fixture'): WeightsSource => ({
  kind: 'directory',
  label,
  bytes: 0,
  model: { files: new Map() },
});
const model = () => ({ dispose: vi.fn() });
const flush = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

function fixture() {
  // Exercise real controller methods, replacing only asynchronous import/model/pass boundaries (no GPU work).
  const demo: any = new DemoController();
  demo.set({
    phase: 'ready',
    backends: [
      { id: 'tsl', label: 'TSL', reason: null },
      { id: 'reference-wgsl', label: 'Reference', reason: null },
    ],
  });
  const builds: ReturnType<typeof deferred<void>>[] = [];
  const resizes: ReturnType<typeof deferred<void>>[] = [];
  demo.pass = {
    setNetwork: vi.fn((builder) => {
      if (!builder) {
        demo.networkStatus('none', 'NR off');
        return Promise.resolve();
      }
      demo.networkStatus('building', 'compiling');
      const build = deferred();
      builds.push(build);
      return build.promise;
    }),
    setSize: vi.fn(() => {
      demo.networkStatus('building', 'resizing');
      const resize = deferred();
      resizes.push(resize);
      return resize.promise;
    }),
    dispose: vi.fn(),
  };
  demo.referenceModule = { loadReferenceModel: vi.fn().mockResolvedValue(model()), referenceWgslBackend: {} };
  vi.spyOn(nr.NRModel, 'load').mockResolvedValue(model() as never);
  weights.read.mockResolvedValue(source());
  weights.synthetic.mockResolvedValue({ ...source(), kind: 'synthetic' });
  return { demo, builds, resizes, ready: () => demo.networkStatus('ready', 'current network ready') };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  weights.read.mockReset();
  weights.synthetic.mockReset();
});

describe('current network readiness', () => {
  it('follows a resolution rebuild after the weight load build resolves superseded', async () => {
    const { demo, builds, resizes, ready } = fixture();
    const load = demo.loadDirectory([]);
    const done = vi.fn();
    const waiting = demo.waitForNetworkReady().then(done);
    await flush();
    expect(builds).toHaveLength(1);
    const resize = demo.setResolution({ width: 640, height: 360 });
    builds[0].resolve();
    await load;
    expect(done).not.toHaveBeenCalled();
    expect(demo.getState().network.state).toBe('building');
    ready();
    resizes[0].resolve();
    await Promise.all([resize, waiting]);
    expect(done).toHaveBeenCalledTimes(1);
    expect(demo.listeners.size).toBe(0);
  });

  it('follows a backend rebuild instead of accepting the superseded load promise', async () => {
    const { demo, builds, ready } = fixture();
    const load = demo.loadDirectory([]);
    await flush();
    const backend = demo.setBackend('reference-wgsl');
    const done = vi.fn();
    const waiting = demo.waitForNetworkReady().then(done);
    await flush();
    expect(builds).toHaveLength(2);
    builds[0].resolve();
    await load;
    expect(done).not.toHaveBeenCalled();
    ready();
    builds[1].resolve();
    await Promise.all([backend, waiting]);
    expect(demo.listeners.size).toBe(0);
  });

  it('ignores an old pass ready notification while uploading the latest weights', async () => {
    const { demo, builds, ready } = fixture();
    demo.set({ weights: { kind: 'directory', label: 'old' }, network: { state: 'ready', message: 'old', since: 0 } });
    const pending = deferred<any>();
    vi.mocked(nr.NRModel.load).mockReturnValueOnce(pending.promise);
    const load = demo.loadDirectory([]);
    await flush();
    const done = vi.fn();
    const waiting = demo.waitForNetworkReady().then(done);
    ready(); // The previous pass can finish while a new model is still being uploaded.
    await flush();
    expect(done).not.toHaveBeenCalled();
    expect(demo.getState().network.state).toBe('building');
    pending.resolve(model());
    await flush();
    ready();
    builds[0].resolve();
    await Promise.all([load, waiting]);
    expect(demo.listeners.size).toBe(0);
  });

  it('resolves an already-ready network and removes subscriptions even with synchronous initial notification', async () => {
    const { demo, ready } = fixture();
    demo.set({ weights: { kind: 'directory', label: 'loaded' } });
    ready();
    const subscribe = demo.subscribe.bind(demo);
    vi.spyOn(demo, 'subscribe').mockImplementation((check: any) => {
      const unsubscribe = subscribe(check);
      check();
      return unsubscribe;
    });
    await demo.waitForNetworkReady();
    expect(demo.listeners.size).toBe(0);
  });

  it('keeps slow compilation as progress without a readiness timeout', async () => {
    vi.useFakeTimers();
    const { demo, builds, ready } = fixture();
    const load = demo.loadDirectory([]);
    await flush();
    const done = vi.fn();
    const waiting = demo.waitForNetworkReady().then(done);
    await vi.advanceTimersByTimeAsync(600_000);
    demo.networkStatus('building', 'still compiling kernels');
    expect(done).not.toHaveBeenCalled();
    expect(demo.getState().network.message).toBe('still compiling kernels');
    ready();
    builds[0].resolve();
    await Promise.all([load, waiting]);
  });

  it.each([
    'weight error',
    'build error',
    'device loss',
    'initialization failure',
    'clear',
    'dispose',
    'abort',
  ] as const)('terminates and unsubscribes on %s', async (reason) => {
    const { demo, builds } = fixture();
    const read = deferred<WeightsSource>();
    if (reason === 'weight error') weights.read.mockReturnValueOnce(read.promise);
    const load = demo.loadDirectory([]);
    await flush();
    const signal = new AbortController();
    const remove = vi.spyOn(signal.signal, 'removeEventListener');
    const waiting = demo.waitForNetworkReady({ signal: signal.signal });
    const rejection = expect(waiting).rejects.toThrow();
    if (reason === 'weight error') read.reject(new Error('bad manifest'));
    if (reason === 'build error') {
      vi.spyOn(console, 'error').mockImplementation(() => {});
      builds[0].reject(new Error('compile failed'));
    }
    if (reason === 'device loss') demo.set({ phase: 'error', error: 'The WebGPU device was lost' });
    if (reason === 'initialization failure') demo.set({ phase: 'error', error: 'WebGPU initialization failed' });
    if (reason === 'clear') await demo.clearWeights();
    if (reason === 'dispose') demo.dispose();
    if (reason === 'abort') signal.abort();
    await rejection;
    expect(demo.listeners.size).toBe(0);
    expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
    builds[0]?.resolve();
    await load;
  });

  it('does not import a cleared read or publish its late progress', async () => {
    const { demo } = fixture();
    const read = deferred<WeightsSource>();
    let progress!: (message: string) => void;
    weights.read.mockImplementationOnce((_files, onProgress) => {
      progress = onProgress;
      return read.promise;
    });
    const load = demo.loadDirectory([]);
    const waiting = demo.waitForNetworkReady();
    const rejection = expect(waiting).rejects.toThrow('cleared');
    await demo.clearWeights();
    await rejection;
    progress('late obsolete progress');
    read.resolve(source('cleared'));
    await load;
    expect(demo.getState().weights).toBeNull();
    expect(demo.getState().weightsBusy).toBeNull();
    expect(nr.NRModel.load).not.toHaveBeenCalled();
  });

  it('retains the newest weights when an earlier read finishes last', async () => {
    const { demo, builds, ready } = fixture();
    const a = deferred<WeightsSource>();
    weights.read.mockReturnValueOnce(a.promise).mockResolvedValueOnce(source('newest'));
    const old = demo.loadDirectory([]);
    const newest = demo.loadDirectory([]);
    await flush();
    const waiting = demo.waitForNetworkReady();
    a.resolve(source('stale'));
    await old;
    expect(demo.getState().weights.label).toBe('newest');
    ready();
    builds[0].resolve();
    await Promise.all([newest, waiting]);
    expect(nr.NRModel.load).toHaveBeenCalledTimes(1);
  });

  it('releases superseded async model results instead of replacing the current prepared model', async () => {
    const { demo, builds, ready } = fixture();
    const oldModel = deferred<any>();
    const stale = model();
    const current = model();
    vi.mocked(nr.NRModel.load)
      .mockReturnValueOnce(oldModel.promise)
      .mockResolvedValueOnce(current as never);
    const a = demo.loadDirectory([]);
    await flush();
    const b = demo.loadDirectory([]);
    await flush();
    const waiting = demo.waitForNetworkReady();
    oldModel.resolve(stale);
    await a;
    expect(stale.dispose).toHaveBeenCalledTimes(1);
    expect(demo.prepared.model).toBe(current);
    ready();
    builds[0].resolve();
    await Promise.all([b, waiting]);
    expect(current.dispose).not.toHaveBeenCalled();
  });

  it('supports the same readiness contract for synthetic weight loading', async () => {
    vi.useFakeTimers();
    const { demo, builds, ready } = fixture();
    const load = demo.loadSyntheticWeights();
    const waiting = demo.waitForNetworkReady();
    await vi.advanceTimersByTimeAsync(30);
    expect(builds).toHaveLength(1);
    ready();
    builds[0].resolve();
    await Promise.all([load, waiting]);
    expect(demo.getState().weights.kind).toBe('synthetic');
  });

  it('rejects already canceled, disposed and empty states without leaving listeners', async () => {
    const { demo } = fixture();
    const signal = new AbortController();
    signal.abort();
    await expect(demo.waitForNetworkReady({ signal: signal.signal })).rejects.toThrow();
    await expect(demo.waitForNetworkReady()).rejects.toThrow('no weights');
    demo.dispose();
    await expect(demo.waitForNetworkReady()).rejects.toThrow('disposed');
    expect(demo.listeners.size).toBe(0);
  });
});
