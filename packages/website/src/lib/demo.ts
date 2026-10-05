// The demo, outside React: the WebGPU device and renderer, the studio and head, the NR pass with a switchable backend,
// and the state the UI renders (subscribe / getState, for useSyncExternalStore).

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as nr from 'three-dlss-nr';
import type {
  DlssNrNetworkBuilder,
  DlssNrNetworkState,
  DlssNrPass,
  DlssNrSettings,
  DlssNrView,
  NRBackendFactory,
  NRBackendId,
} from 'three-dlss-nr';

import { DEFAULT_MODEL_ID, headModel } from './models';
import { createStudio, loadHead, type LoadedHead, type Studio } from './studio';
import { readModelDirectory, syntheticWeights, type WeightsSource } from './weights';

export type FrameMode = 'shared' | 'reference';

export interface BackendOption {
  id: NRBackendId;
  label: string;
  /** Null when it can run here; otherwise a short reason for the UI ... */
  reason: string | null;
  /** ... and the backend's full explanation (how to fix it). */
  detail: string | null;
}

export interface DeviceReport {
  adapter: string;
  shaderF16: boolean;
  timestampQuery: boolean;
  limits: { name: string; value: number }[];
  /** The device could not be configured for the network (NR unavailable; the scene still renders). */
  problem: string | null;
}

export interface BackendTiming {
  /** Median GPU ms of the last frames (timestamp queries), or null when unavailable. */
  gpuMs: number | null;
  /** Median wall ms (submit to GPU done) of the network. */
  wallMs: number;
  frames: number;
}

export interface Resolution {
  width: number;
  height: number;
}

export const RESOLUTIONS: readonly Resolution[] = [
  { width: 640, height: 360 },
  { width: 960, height: 540 },
  { width: 1280, height: 720 },
];

export interface DemoState {
  phase: 'starting' | 'ready' | 'error';
  /** Heading and message of a fatal error (no WebGPU, device lost). */
  errorTitle: string | null;
  error: string | null;
  device: DeviceReport | null;
  backends: BackendOption[];
  backendId: NRBackendId;
  frameMode: FrameMode;
  weights: { kind: WeightsSource['kind']; label: string } | null;
  /** A weights operation in progress (reading a directory, generating synthetic weights, uploading). */
  weightsBusy: string | null;
  weightsError: string | null;
  network: { state: DlssNrNetworkState; message: string; since: number };
  view: DlssNrView;
  split: number;
  settings: Required<DlssNrSettings>;
  resolution: Resolution;
  modelId: string;
  modelLoading: boolean;
  autoRotate: boolean;
  timings: Partial<Record<NRBackendId, BackendTiming>>;
  fps: number;
}

type Listener = () => void;

const BACKEND_LABELS: Record<NRBackendId, string> = {
  tsl: 'TSL (native three.js)',
  'reference-wgsl': 'Reference WGSL (OpenDLSS-NR)',
};

const median = (values: number[]): number | null => nr.summarizeMilliseconds(values)?.median ?? null;

interface Prepared {
  backendId: NRBackendId;
  weights: WeightsSource;
  model: unknown;
  dispose(): void;
}

export class DemoController {
  private state: DemoState;
  private listeners = new Set<Listener>();
  private renderer: any = null;
  private camera: any = null;
  private controls: any = null;
  private studio: Studio | null = null;
  private head: LoadedHead | null = null;
  private pass: DlssNrPass | null = null;
  private referenceModule: typeof import('three-dlss-nr/reference-backend') | null = null;
  private weightsSource: WeightsSource | null = null;
  private prepared: Prepared | null = null;
  private samples = new Map<NRBackendId, { gpu: number[]; wall: number[]; frames: number }>();
  private frameTimes: number[] = [];
  private lastEmit = 0;
  private resizeObserver: ResizeObserver | null = null;
  private container: HTMLElement | null = null;
  private disposed = false;
  private networkRequest = 0;
  private weightsRequest = 0;
  private networkPreparing = false;
  private lifetime = new AbortController();

  constructor() {
    this.state = {
      phase: 'starting',
      errorTitle: null,
      error: null,
      device: null,
      backends: (['tsl', 'reference-wgsl'] as const).map((id) => ({
        id,
        label: BACKEND_LABELS[id],
        reason: 'checking the device',
        detail: null,
      })),
      backendId: 'tsl',
      frameMode: 'shared',
      weights: null,
      weightsBusy: null,
      weightsError: null,
      network: { state: 'none', message: 'NR off: no weights loaded', since: performance.now() },
      view: 'off',
      split: 0.5,
      settings: { ...nr.DLSS_NR_DEFAULT_SETTINGS },
      resolution: RESOLUTIONS[1],
      modelId: DEFAULT_MODEL_ID,
      modelLoading: true,
      autoRotate: true,
      timings: {},
      fps: 0,
    };
  }

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): DemoState => this.state;

  private set(patch: Partial<DemoState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }

  private networkStatus = (state: DlssNrNetworkState, message: string): void => {
    // While the current request uploads weights, the pass may still report a previous build's status.
    if (this.disposed || this.networkPreparing) return;
    this.set({
      network: {
        state,
        message,
        since: state === this.state.network.state ? this.state.network.since : performance.now(),
      },
    });
  };

  /**
   * Wait for the current weights/backend/resolution to reach ready, following superseding rebuilds. Weight import
   * methods complete their own attempt; they do not guarantee that a newer build is ready. Call this afterwards
   * when readiness is required. Compilation has no timeout: progress stays pending until ready, failure, clearing
   * weights, disposal or caller cancellation. An AbortSignal can bound a caller's wait without canceling the build.
   */
  waitForNetworkReady({ signal }: { signal?: AbortSignal } = {}): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      let unsubscribe: (() => void) | undefined;
      const finish = (error?: unknown) => {
        if (settled) return;
        settled = true;
        unsubscribe?.();
        signal?.removeEventListener('abort', canceled);
        this.lifetime.signal.removeEventListener('abort', disposed);
        if (error !== undefined) reject(error);
        else resolve();
      };
      const canceled = () =>
        finish(signal?.reason ?? new DOMException('Network readiness wait canceled', 'AbortError'));
      const disposed = () => finish(new Error('Demo controller disposed'));
      const check = () => {
        if (this.disposed) return disposed();
        if (signal?.aborted) return canceled();
        const state = this.state;
        if (state.phase === 'error') return finish(new Error(state.error ?? 'Demo initialization failed'));
        if (state.weightsError) return finish(new Error(state.weightsError));
        if (!state.weights && !state.weightsBusy)
          return finish(new Error('Network has no weights; load was cleared or canceled'));
        if (state.weightsBusy || this.networkPreparing) return;
        if (state.network.state === 'failed') return finish(new Error(state.network.message));
        if (state.network.state === 'ready') finish();
      };
      signal?.addEventListener('abort', canceled, { once: true });
      this.lifetime.signal.addEventListener('abort', disposed, { once: true });
      unsubscribe = this.subscribe(check);
      // subscribe currently does not notify immediately, but a synchronous initial callback is safe too.
      if (settled) unsubscribe();
      else check();
    });
  }

  /** Create the renderer in `container` and start rendering. */
  async start(container: HTMLElement): Promise<void> {
    this.container = container;
    try {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error(
          'This browser does not expose WebGPU (navigator.gpu is missing). Use a current Chrome or Edge on Windows, ' +
            'macOS or ChromeOS, or Safari 26+; on Linux, Chrome needs WebGPU enabled in chrome://flags.',
        );
      }
      let device: GPUDevice | undefined;
      let adapter: GPUAdapter | undefined;
      let problem: string | null = null;
      try {
        const created = await nr.createNRDevice();
        device = created.device;
        adapter = created.adapter;
      } catch (error) {
        problem = error instanceof Error ? error.message : String(error);
      }
      const renderer = new THREE.WebGPURenderer(device ? { device, antialias: false } : { antialias: false });
      await renderer.init();
      if (!renderer.backend?.device) throw new Error('WebGPU initialisation failed: three.js fell back to WebGL.');
      if (this.disposed) {
        renderer.dispose();
        return;
      }
      this.renderer = renderer;
      void (renderer.backend.device as GPUDevice).lost.then((info) => {
        if (this.disposed || info.reason === 'destroyed') return;
        renderer.setAnimationLoop(null);
        this.set({
          phase: 'error',
          errorTitle: 'The GPU stopped responding',
          error:
            `The WebGPU device was lost (${info.message || info.reason}). This happens when the GPU is busy with ` +
            'other work or one frame takes too long (Windows resets the GPU after about two seconds). Reload the ' +
            'page and try a lower resolution.',
        });
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.style.display = 'block';
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      container.appendChild(renderer.domElement);

      this.set({ device: deviceReport(renderer.backend.device, adapter, problem) });
      this.referenceModule = await import('three-dlss-nr/reference-backend');
      this.set({ backends: this.backendOptions(problem) });
      const firstAvailable = this.state.backends.find((option) => option.reason === null);
      if (firstAvailable) this.set({ backendId: firstAvailable.id });

      this.studio = createStudio(renderer);
      this.camera = new THREE.PerspectiveCamera(26, 16 / 9, 0.1, 100);
      this.camera.position.set(1.2, 0.25, 5.4);
      this.controls = new OrbitControls(this.camera, renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.autoRotate = this.state.autoRotate;
      this.controls.autoRotateSpeed = 0.6;
      this.controls.minDistance = 2.5;
      this.controls.maxDistance = 12;
      this.controls.target.set(0, 0.05, 0);
      this.controls.update();

      const { width, height } = this.state.resolution;
      this.pass = new nr.DlssNrPass({
        renderer,
        scene: this.studio.scene,
        camera: this.camera,
        width,
        height,
        view: this.state.view,
        settings: this.state.settings,
      });
      this.pass.onStatus = this.networkStatus;

      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(container);
      this.resize();
      renderer.setAnimationLoop(() => this.frame());
      this.set({ phase: 'ready' });
      await this.setModel(this.state.modelId);
    } catch (error) {
      this.set({
        phase: 'error',
        errorTitle: 'This demo needs WebGPU',
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private backendOptions(deviceProblem: string | null): BackendOption[] {
    const renderer = this.renderer;
    const option = (id: NRBackendId, factory: NRBackendFactory | undefined, missing: string): BackendOption => {
      if (deviceProblem)
        return { id, label: BACKEND_LABELS[id], reason: 'device limits too low', detail: deviceProblem };
      if (!factory) return { id, label: BACKEND_LABELS[id], reason: missing, detail: missing };
      const detail = factory.unavailableReason(renderer);
      if (!detail) return { id, label: BACKEND_LABELS[id], reason: null, detail: null };
      const device: GPUDevice = renderer.backend.device;
      const reason = factory.requirements.features.some((feature) => !device.features.has(feature))
        ? `needs ${factory.requirements.features.map((f) => `'${f}'`).join(', ')}, which this GPU / browser lacks`
        : 'device limits too low';
      return { id, label: BACKEND_LABELS[id], reason, detail };
    };
    return [
      option('tsl', nr.tslBackend, 'the TSL backend failed to load'),
      option('reference-wgsl', this.referenceModule?.referenceWgslBackend, 'reference backend failed to load'),
    ];
  }

  private resize(): void {
    const container = this.container;
    if (!container || !this.renderer) return;
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private frame(): void {
    const pass = this.pass;
    if (!pass) return;
    this.controls.update();
    void pass.render().then((stats) => {
      if (!stats) return;
      const now = performance.now();
      this.frameTimes.push(now);
      while (this.frameTimes.length > 30) this.frameTimes.shift();
      if (stats.network) {
        const id = pass.backend?.id ?? this.state.backendId;
        const sample = this.samples.get(id) ?? { gpu: [], wall: [], frames: 0 };
        if (stats.network.gpuMilliseconds !== null) sample.gpu.push(stats.network.gpuMilliseconds);
        sample.wall.push(stats.network.wallMilliseconds);
        sample.frames += 1;
        while (sample.gpu.length > 30) sample.gpu.shift();
        while (sample.wall.length > 30) sample.wall.shift();
        this.samples.set(id, sample);
      }
      if (now - this.lastEmit > 250) {
        this.lastEmit = now;
        const timings: DemoState['timings'] = {};
        for (const [id, sample] of this.samples) {
          timings[id] = {
            gpuMs: median(sample.gpu),
            wallMs: median(sample.wall) ?? 0,
            frames: sample.frames,
          };
        }
        const span = this.frameTimes.length > 1 ? this.frameTimes.at(-1)! - this.frameTimes[0] : 0;
        const fps = span > 0 ? ((this.frameTimes.length - 1) * 1000) / span : 0;
        this.set({ timings, fps });
      }
    });
  }

  // ---------------------------------------------------------------------------------------------------------------
  // UI actions
  // ---------------------------------------------------------------------------------------------------------------

  setView(view: DlssNrView): void {
    if (this.pass) this.pass.view = view;
    this.set({ view });
  }

  setSplit(split: number): void {
    const clamped = Math.min(1, Math.max(0, split));
    if (this.pass) this.pass.split = clamped;
    this.set({ split: clamped });
  }

  setSettings(settings: DlssNrSettings): void {
    this.pass?.setSettings(settings);
    this.set({ settings: { ...this.state.settings, ...settings } });
  }

  setAutoRotate(autoRotate: boolean): void {
    if (this.controls) this.controls.autoRotate = autoRotate;
    this.set({ autoRotate });
  }

  resetCamera(): void {
    this.camera?.position.set(1.2, 0.25, 5.4);
    this.controls?.target.set(0, 0.05, 0);
    this.controls?.update();
    this.pass?.resetHistory();
  }

  resetHistory(): void {
    this.pass?.resetHistory();
  }

  async setResolution(resolution: Resolution): Promise<void> {
    this.set({ resolution });
    await this.pass?.setSize(resolution.width, resolution.height).catch(() => undefined);
  }

  async setModel(id: string): Promise<void> {
    const entry = headModel(id);
    this.set({ modelId: id, modelLoading: true });
    try {
      const head = await loadHead(entry);
      if (this.disposed || this.state.modelId !== id) {
        head.dispose();
        return;
      }
      if (this.head) {
        this.studio!.stage.remove(this.head.object);
        this.head.dispose();
      }
      this.head = head;
      this.studio!.stage.add(head.object);
      // A new subject is a camera cut for the temporal history.
      this.pass?.resetHistory();
    } finally {
      if (this.state.modelId === id) this.set({ modelLoading: false });
    }
  }

  async setBackend(backendId: NRBackendId): Promise<void> {
    this.set({ backendId });
    await this.applyNetwork();
  }

  async setFrameMode(frameMode: FrameMode): Promise<void> {
    this.set({ frameMode });
    if (this.state.backendId === 'reference-wgsl') await this.applyNetwork();
  }

  /**
   * Import a local directory and attempt its build. Inspect weightsError/network for failure; use
   * waitForNetworkReady afterwards to await the current network if this build was superseded.
   */
  async loadDirectory(files: FileList | readonly File[]): Promise<void> {
    await this.loadWeights((onProgress) => readModelDirectory(files, onProgress));
  }

  /** Generate/import synthetic weights and attempt their build; current readiness is a separate wait. */
  async loadSyntheticWeights(): Promise<void> {
    await this.loadWeights(async (onProgress) => {
      onProgress('generating synthetic weights (about 141 MiB)…');
      // Let the message paint before the generator blocks the main thread for a second or two.
      await new Promise((done) => setTimeout(done, 30));
      return syntheticWeights();
    });
  }

  /** Drop the weights: NR off. */
  async clearWeights(): Promise<void> {
    ++this.weightsRequest;
    this.weightsSource = null;
    this.set({ weights: null, weightsBusy: null, weightsError: null, view: 'off' });
    if (this.pass) this.pass.view = 'off';
    await this.applyNetwork();
  }

  private async loadWeights(read: (onProgress: (message: string) => void) => Promise<WeightsSource>): Promise<void> {
    if (this.disposed) return;
    const request = ++this.weightsRequest;
    this.set({ weightsBusy: 'reading weights…', weightsError: null });
    try {
      const source = await read((message) => {
        if (!this.disposed && request === this.weightsRequest) this.set({ weightsBusy: message });
      });
      if (this.disposed || request !== this.weightsRequest) return;
      this.weightsSource = source;
      this.set({
        weights: { kind: source.kind, label: source.label },
        weightsBusy: null,
        network: { state: 'building', message: 'preparing the network', since: performance.now() },
      });
      if (this.state.view === 'off') this.setView('split');
      await this.applyNetwork();
    } catch (error) {
      if (this.disposed || request !== this.weightsRequest) return;
      this.set({ weightsBusy: null, weightsError: error instanceof Error ? error.message : String(error) });
    }
  }

  /** (Re)build the pass's network for the current backend, frame mode and weights. */
  private async applyNetwork(): Promise<void> {
    const pass = this.pass;
    if (!pass || this.disposed) return;
    const request = ++this.networkRequest;
    this.networkPreparing = true;
    const weights = this.weightsSource;
    const option = this.state.backends.find((candidate) => candidate.id === this.state.backendId);
    if (!weights || !option || option.reason !== null) {
      await pass.setNetwork(null);
      if (this.disposed || request !== this.networkRequest) return;
      this.releasePrepared();
      this.networkPreparing = false;
      if (weights && option?.reason) {
        this.set({
          network: { state: 'failed', message: `${option.label}: ${option.reason}`, since: performance.now() },
        });
      } else this.set({ network: { state: 'none', message: 'NR off: no weights loaded', since: performance.now() } });
      return;
    }
    this.set({ network: { state: 'building', message: 'preparing the network', since: performance.now() } });
    try {
      const builder = await this.networkBuilder(option.id, weights, request);
      if (this.disposed || request !== this.networkRequest || !builder) return;
      this.networkPreparing = false;
      await pass.setNetwork(builder);
    } catch (error) {
      if (this.disposed || request !== this.networkRequest) return;
      this.networkPreparing = false;
      const message = error instanceof Error ? error.message : String(error);
      console.error(error);
      this.set({ network: { state: 'failed', message, since: performance.now() } });
    }
  }

  private releasePrepared(): void {
    this.prepared?.dispose();
    this.prepared = null;
  }

  /** Load the weights for `backendId` once (kept across resizes and frame-mode changes) and return a builder. */
  private async networkBuilder(
    backendId: NRBackendId,
    weights: WeightsSource,
    request: number,
  ): Promise<DlssNrNetworkBuilder | null> {
    if (this.prepared && (this.prepared.backendId !== backendId || this.prepared.weights !== weights)) {
      // Switching backends: the other backend's copy of the weights is no longer needed (the old network goes with
      // the next setNetwork; it does not own the shared model).
      await this.pass!.setNetwork(null);
      if (this.disposed || request !== this.networkRequest) return null;
      this.releasePrepared();
    }
    const reference = this.referenceModule!;
    if (!this.prepared) {
      this.set({ network: { state: 'building', message: 'uploading weights', since: performance.now() } });
      if (backendId === 'reference-wgsl') {
        const model = await reference.loadReferenceModel(this.renderer, weights.model, (message) => {
          if (!this.disposed && request === this.networkRequest)
            this.set({ network: { ...this.state.network, message } });
        });
        if (this.disposed || request !== this.networkRequest) {
          model.dispose();
          return null;
        }
        this.prepared = { backendId, weights, model, dispose: () => model.dispose() };
      } else {
        // Parsed once and borrowed by every network (resizes do not reload it).
        const model = await nr.NRModel.load(weights.model);
        if (this.disposed || request !== this.networkRequest) {
          model.dispose();
          return null;
        }
        this.prepared = { backendId, weights, model, dispose: () => model.dispose() };
      }
    }
    const model = this.prepared.model;
    if (backendId === 'reference-wgsl') {
      if (this.state.frameMode === 'reference') {
        return ({ renderer, width, height, onProgress }) =>
          reference.ReferenceFrame.create({ renderer, model: model as never, width, height, onProgress });
      }
      return nr.backendBuilder(reference.referenceWgslBackend, model as object);
    }
    return nr.backendBuilder(nr.tslBackend, model as object);
  }

  dispose(): void {
    this.disposed = true;
    ++this.weightsRequest;
    ++this.networkRequest;
    this.lifetime.abort();
    this.resizeObserver?.disconnect();
    this.renderer?.setAnimationLoop(null);
    this.pass?.dispose();
    this.releasePrepared();
    this.head?.dispose();
    this.studio?.dispose();
    this.controls?.dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }
}

function deviceReport(device: GPUDevice, adapter: GPUAdapter | undefined, problem: string | null): DeviceReport {
  const info = (adapter as (GPUAdapter & { info?: GPUAdapterInfo }) | undefined)?.info ?? (device as any).adapterInfo;
  const adapterName = info
    ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'unknown adapter'
    : 'unknown adapter';
  const limits = device.limits as unknown as Record<string, number>;
  return {
    adapter: adapterName,
    shaderF16: device.features.has('shader-f16'),
    timestampQuery: device.features.has('timestamp-query'),
    limits: [
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxStorageBuffersPerShaderStage',
      'maxStorageBufferBindingSize',
      'maxBufferSize',
    ].map((name) => ({ name, value: limits[name] })),
    problem,
  };
}
