// A drop-in render pass: a three.js scene in, the neural-rendered frame out, with NR on, off or split-screen.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The frame around the network follows the
// reference's demo (ports/browser-webgpu/demo/src/backend/production-pipeline.js): input features, the network,
// compose with the history, the display transform. What differs is how a frame arrives (design section 4.4): the scene
// is rendered by three into an MRT target (linear HDR colour + three's `velocity`), and the frame kernels read those
// textures on the GPU, with no readback.
//
// One frame (`render()`):
//   1. the scene into `renderTarget` (RGBA16F colour, RGBA16F velocity), `mrt({ output, velocity })`;
//   2. with a network and a view that shows it: `input_features` (writes `backend.features` in place), the backend's
//      `run` (any `NRBackend`: the native TSL port or the reference WGSL shim; both share the feature and head
//      layout), `compose` (reads `backend.head`, writes the next history and the reference's bgra8 image), and an
//      unpack of that image into `outputTexture` (display-ready sRGB bytes);
//   3. a full-screen quad onto the canvas: NR off = the reference's display transform (`nrDisplayTransform`, the ACES
//      fit + sRGB of frame.wgsl) of the scene; NR on = `outputTexture`; split = off left of `split`, on right of it.
//
// The frame kernels are chunk D's (`createFrameKernels`), bit-identical to the reference's frame.wgsl on D3D12, so
// both backends share them. A backend can also bring its own frame (`DlssNrExternalFrame`, e.g. the reference shim's
// `ReferenceFrame`, the upstream frame recorded into the upstream graph), which this pass then drives instead.

import { HalfFloatType, NoToneMapping, LinearSRGBColorSpace, Color } from 'three';
import { NodeMaterial, QuadMesh, RenderTarget, StorageTexture } from 'three/webgpu';
import {
  Fn,
  If,
  float,
  localId,
  mrt,
  output,
  screenUV,
  select,
  storage,
  texture,
  textureStore,
  uniform,
  uvec2,
  vec4,
  velocity,
  workgroupId,
} from 'three/tsl';

import type { NRBackend, NRBackendFactory, NRBackendModelSource, NRFrameTiming } from '../backend/NRBackend.js';
import { nrDisplayTransform } from '../frame/compose.js';
import { NRFrameParams } from '../frame/frameInputs.js';
import { createFrameKernels, NRHistory, type FrameKernels } from '../frame/history.js';
import { wordAttribute } from '../tensors.js';
import type { BufferSource } from '../tsl/KernelBuilder.js';
import { u, type TSLNode } from '../tsl/packed.js';

/** What the canvas shows. */
export type DlssNrView = 'off' | 'on' | 'split';

/** The reference demo's NR controls (`demo/src/backend/nr-settings.js`), plus the display parameters. */
export interface DlssNrSettings {
  /** 1 = the full correction; lower dials it back towards the rendered frame. Default 1. */
  intensity?: number;
  /** 0 none, 1 cinematic, 2 natural (a grade on the network's output). Default 0. */
  style?: number;
  /** Strength of the style's local tone. Default 1. */
  localTone?: number;
  localStructure?: number;
  skinStructure?: number;
  autoMask?: boolean;
  /** Scene value that maps to display white. Default 1. */
  paperWhite?: number;
  /** 0 = the network changes luminance only, 1 = colour too. Default 1. */
  colorStrength?: number;
}

/** The reference demo's defaults (`NR_DEFAULTS`). */
export const DLSS_NR_DEFAULT_SETTINGS: Readonly<Required<DlssNrSettings>> = {
  intensity: 1,
  style: 0,
  localTone: 1,
  localStructure: 1,
  skinStructure: -1,
  autoMask: true,
  paperWhite: 1,
  colorStrength: 1,
};

/**
 * A network that brings its own frame (input features, compose, history, display): the pass hands it the rendered
 * textures and draws its `output`. The reference shim's `ReferenceFrame` has this shape.
 */
export interface DlssNrExternalFrame {
  readonly backend: NRBackend;
  /** Display-ready image: an rgba8unorm texture holding sRGB-encoded bytes, `width x height`. */
  readonly output: any;
  paperWhite: number;
  colorStrength: number;
  render(input: {
    color: any;
    velocity?: any;
    reset?: boolean;
    settings?: Omit<DlssNrSettings, 'paperWhite' | 'colorStrength'> & { enabled?: boolean };
    timing?: boolean;
  }): Promise<NRFrameTiming>;
  resetHistory(): void;
  dispose(): void;
}

/** What `DlssNrNetworkBuilder` returns: a bare backend (the pass runs the frame) or a backend with its own frame. */
export type DlssNrNetwork = NRBackend | DlssNrExternalFrame;

/** Builds the network at a size; called again (after disposing the old one) when the pass is resized. */
export type DlssNrNetworkBuilder = (options: {
  renderer: any;
  width: number;
  height: number;
  onProgress?: (message: string) => void;
}) => Promise<DlssNrNetwork>;

/** A builder that creates `factory`'s backend with `model` (load the weights once and pass the loaded model). */
export function backendBuilder(
  factory: NRBackendFactory,
  model: NRBackendModelSource | object,
  extra: Record<string, unknown> = {},
): DlssNrNetworkBuilder {
  return ({ renderer, width, height, onProgress }) =>
    factory.create({ ...extra, renderer, width, height, onProgress, model: model as NRBackendModelSource });
}

const isExternalFrame = (network: DlssNrNetwork): network is DlssNrExternalFrame =>
  'backend' in network && typeof (network as DlssNrExternalFrame).render === 'function';

/**
 * The model's learned cap on the history blend weight, which compose needs: `blendScale` on the reference shim, the
 * model's `blendScale()` on the TSL port, else `fallback`.
 */
export function backendBlendScale(backend: NRBackend, fallback?: number): number {
  const candidate = backend as unknown as { blendScale?: unknown; model?: { blendScale?: () => number } };
  if (typeof candidate.blendScale === 'number') return candidate.blendScale;
  if (typeof candidate.blendScale === 'function') return (candidate.blendScale as () => number).call(backend);
  if (typeof candidate.model?.blendScale === 'function') return candidate.model.blendScale();
  if (fallback !== undefined) return fallback;
  throw new Error(`cannot find the blend scale of the ${backend.id} backend; pass \`blendScale\` to the pass`);
}

/** Timing and state of one `render()`. */
export interface DlssNrFrameStats {
  /** The network ran this frame. */
  ran: boolean;
  /** The backend's own timing of the network (`run`), or the external frame's timing of its whole frame. */
  network: NRFrameTiming | null;
  /** Wall time of the NR part of the frame (features, network, compose), ms. */
  nrWallMilliseconds: number | null;
  /** The frame started without history (first frame, reset, rebuild). */
  historyReset: boolean;
}

export interface DlssNrPassOptions {
  /** An initialized `WebGPURenderer` (on a device from `createNRDevice` when the reference backend may run). */
  renderer: any;
  scene: any;
  camera: any;
  /** Internal (network) resolution; the canvas shows it scaled. Default 960x540. */
  width?: number;
  height?: number;
  settings?: DlssNrSettings;
  view?: DlssNrView;
  /** Measure the network's GPU time with timestamp queries (default true when the device has them). */
  timing?: boolean;
  /**
   * Per-pixel history rejection when the reprojected position leaves the screen (the native pipeline's history flag;
   * the reference WebGPU port does not do it). Default false, as the reference.
   */
  rejectOffscreenHistory?: boolean;
  /** Blend scale to use when the backend does not expose one (see `backendBlendScale`). */
  blendScale?: number;
}

/** Status of the network slot. */
export type DlssNrNetworkState = 'none' | 'building' | 'ready' | 'failed';

/** Largest valid size the pass builds a network for (design 4.2: cap the demo at 1280x720). */
export const DLSS_NR_MAX_PIXELS = 1280 * 720;

/**
 * Renders `scene` from `camera` through the NR network and presents the result on the renderer's canvas.
 *
 * ```ts
 * const pass = new DlssNrPass({ renderer, scene, camera, width: 960, height: 540 });
 * await pass.setNetwork(backendBuilder(tslBackend, model));
 * renderer.setAnimationLoop(async () => { await pass.render(); });
 * ```
 *
 * The pass sets the renderer's tone mapping to none and its output colour space to linear: the quad it draws already
 * holds display-ready (sRGB-encoded) values, from the reference's display transform.
 */
export class DlssNrPass {
  readonly renderer: any;
  scene: any;
  camera: any;
  /** The MRT target the scene is rendered into: `textures[0]` colour (linear HDR), `textures[1]` velocity. */
  readonly renderTarget: any;
  view: DlssNrView;
  /** Split position in [0, 1] of the canvas width (split view: NR off left, NR on right). */
  split = 0.5;
  /** Measure GPU time of the network with timestamp queries. */
  timing: boolean;
  /** Called with progress / state messages (network builds, rebuilds after a resize, failures). */
  onStatus: ((state: DlssNrNetworkState, message: string) => void) | null = null;
  private settingsValue: Required<DlssNrSettings>;
  private readonly mrtNode: any;
  private readonly quad: any;
  private readonly splitUniform = uniform(0.5);
  /** 0 = off, 1 = on, 2 = split. */
  private readonly viewUniform = uniform(0, 'uint');
  private readonly noOutput: any;
  private readonly outputSample: any;
  private readonly rejectOffscreenHistory: boolean;
  private readonly fallbackBlendScale: number | undefined;
  private size: { width: number; height: number };
  private builder: DlssNrNetworkBuilder | null = null;
  private network: DlssNrNetwork | null = null;
  private frame: TslFrame | null = null;
  private generation = 0;
  private stateValue: DlssNrNetworkState = 'none';
  private messageValue = '';
  private historyValid = false;
  private frameIndex = 0;
  private busy = false;
  private disposed = false;

  constructor(options: DlssNrPassOptions) {
    const { renderer } = options;
    if (!renderer?.backend?.device) {
      throw new Error('DlssNrPass needs an initialized WebGPURenderer on the WebGPU backend (`await renderer.init()`)');
    }
    this.renderer = renderer;
    this.scene = options.scene;
    this.camera = options.camera;
    this.size = clampSize(options.width ?? 960, options.height ?? 540);
    this.settingsValue = { ...DLSS_NR_DEFAULT_SETTINGS, ...options.settings };
    this.view = options.view ?? 'on';
    this.timing = options.timing ?? renderer.backend.device.features.has('timestamp-query');
    this.rejectOffscreenHistory = options.rejectOffscreenHistory ?? false;
    this.fallbackBlendScale = options.blendScale;

    const { width, height } = this.size;
    this.renderTarget = new RenderTarget(width, height, {
      count: 2,
      type: HalfFloatType,
      depthBuffer: true,
      samples: 0,
    });
    this.renderTarget.textures[0].name = 'output';
    this.renderTarget.textures[1].name = 'velocity';
    this.mrtNode = mrt({ output, velocity });
    // The background has no motion.
    this.mrtNode.setClearColor('velocity', new Color(0, 0, 0), 0);

    // The quad: the display transform of the scene (NR off) and the NR output (display-ready), by view and split.
    this.noOutput = new StorageTexture(1, 1);
    this.outputSample = texture(this.noOutput, screenUV);
    const material = new NodeMaterial();
    material.name = 'three-dlss-nr present';
    material.toneMapped = false;
    material.depthTest = false;
    material.depthWrite = false;
    material.fragmentNode = Fn(() => {
      const uv = screenUV;
      const off = nrDisplayTransform(texture(this.renderTarget.textures[0], uv).xyz.max(float(0)));
      const on = this.outputSample.xyz;
      const showOn = select(
        this.viewUniform.equal(u(2)),
        uv.x.greaterThanEqual(this.splitUniform),
        this.viewUniform.equal(u(1)),
      );
      return vec4(select(showOn, on, off), float(1));
    })();
    this.quad = new QuadMesh(material);
  }

  /** The internal (network) resolution. */
  get width(): number {
    return this.size.width;
  }

  get height(): number {
    return this.size.height;
  }

  get settings(): Readonly<Required<DlssNrSettings>> {
    return this.settingsValue;
  }

  /** Update some settings (takes effect on the next frame). */
  setSettings(settings: DlssNrSettings): void {
    this.settingsValue = { ...this.settingsValue, ...settings };
  }

  /** The network slot's state and its last message. */
  get state(): DlssNrNetworkState {
    return this.stateValue;
  }

  get message(): string {
    return this.messageValue;
  }

  /** The current backend (null without a network, or while it builds). */
  get backend(): NRBackend | null {
    if (!this.network || this.stateValue !== 'ready') return null;
    return isExternalFrame(this.network) ? this.network.backend : this.network;
  }

  /** The display-ready NR image (rgba8unorm, sRGB bytes) of the last NR frame, or null. */
  get outputTexture(): any {
    if (!this.network || this.stateValue !== 'ready') return null;
    return isExternalFrame(this.network) ? this.network.output : this.frame?.output;
  }

  /** Start the next frame without history (a camera cut, a model switch). */
  resetHistory(): void {
    this.historyValid = false;
    if (this.network && isExternalFrame(this.network)) this.network.resetHistory();
  }

  /**
   * Use `builder`'s network (or none). Disposes the current one, builds the new one at the current size and resolves
   * when this build finishes (rejects if the current build fails; the pass then shows NR off). A build superseded
   * by setNetwork, setSize or disposal resolves after releasing its result, without guaranteeing the current
   * network is ready. Observe onStatus/state for current readiness. The history starts over.
   */
  async setNetwork(builder: DlssNrNetworkBuilder | null): Promise<void> {
    this.builder = builder;
    await this.rebuild();
  }

  /**
   * Change the internal resolution (clamped to `DLSS_NR_MAX_PIXELS`, aspect kept). The network is rebuilt at the new
   * size (its geometry is baked into its kernels); until then the pass shows NR off.
   */
  async setSize(width: number, height: number): Promise<void> {
    const size = clampSize(width, height);
    if (size.width === this.size.width && size.height === this.size.height) return;
    this.size = size;
    this.renderTarget.setSize(size.width, size.height);
    if (this.builder) await this.rebuild();
  }

  /**
   * One frame: render the scene, run NR when the view shows it, present to the canvas. Resolves when the network's
   * GPU work is done (it waits on the GPU). A call while the previous one is still running is skipped (returns null).
   */
  async render(): Promise<DlssNrFrameStats | null> {
    if (this.disposed || this.busy) return null;
    this.busy = true;
    try {
      const renderer = this.renderer;
      const previousMRT = renderer.getMRT();
      const previousTarget = renderer.getRenderTarget();
      renderer.setMRT(this.mrtNode);
      renderer.setRenderTarget(this.renderTarget);
      renderer.render(this.scene, this.camera);
      renderer.setRenderTarget(previousTarget);
      renderer.setMRT(previousMRT);

      const stats: DlssNrFrameStats = { ran: false, network: null, nrWallMilliseconds: null, historyReset: false };
      const ready = this.network !== null && this.stateValue === 'ready';
      const runNetwork = ready && this.view !== 'off';
      if (runNetwork) {
        const started = performance.now();
        stats.historyReset = !this.historyValid;
        stats.network = await this.runNetwork();
        stats.nrWallMilliseconds = performance.now() - started;
        stats.ran = true;
      } else {
        // Frames without NR break the temporal chain.
        this.historyValid = false;
      }
      this.present(runNetwork ? this.view : 'off');
      return stats;
    } finally {
      this.busy = false;
    }
  }

  /** Release the network, the render target and the quad (not the renderer). */
  dispose(): void {
    this.disposed = true;
    this.generation += 1;
    this.releaseNetwork();
    this.renderTarget.dispose();
    this.quad.material.dispose();
    this.noOutput.dispose();
  }

  private async runNetwork(): Promise<NRFrameTiming> {
    const network = this.network!;
    const s = this.settingsValue;
    const reset = !this.historyValid;
    let timing: NRFrameTiming;
    if (isExternalFrame(network)) {
      network.paperWhite = s.paperWhite;
      network.colorStrength = s.colorStrength;
      timing = await network.render({
        color: this.renderTarget.textures[0],
        velocity: this.renderTarget.textures[1],
        reset,
        settings: {
          enabled: true,
          intensity: s.intensity,
          style: s.style,
          localTone: s.localTone,
          localStructure: s.localStructure,
          skinStructure: s.skinStructure,
          autoMask: s.autoMask,
        },
        timing: this.timing,
      });
    } else {
      const frame = this.frame!;
      const parity = (this.frameIndex & 1) as 0 | 1;
      frame.params.set({ ...s, seed: this.frameIndex & 0xffff, historyValid: !reset, enabled: true });
      this.renderer.compute(frame.kernels.inputFeatures[parity].node);
      timing = await network.run({ timing: this.timing });
      this.renderer.compute([frame.kernels.compose[parity].node, frame.unpack]);
    }
    this.frameIndex += 1;
    this.historyValid = true;
    return timing;
  }

  private present(view: DlssNrView): void {
    this.viewUniform.value = view === 'on' ? 1 : view === 'split' ? 2 : 0;
    this.splitUniform.value = Math.min(1, Math.max(0, this.split));
    this.outputSample.value = this.outputTexture ?? this.noOutput;
    const renderer = this.renderer;
    const toneMapping = renderer.toneMapping;
    const colorSpace = renderer.outputColorSpace;
    renderer.toneMapping = NoToneMapping;
    renderer.outputColorSpace = LinearSRGBColorSpace;
    this.quad.render(renderer);
    renderer.toneMapping = toneMapping;
    renderer.outputColorSpace = colorSpace;
  }

  private setState(state: DlssNrNetworkState, message: string): void {
    this.stateValue = state;
    this.messageValue = message;
    this.onStatus?.(state, message);
  }

  private releaseNetwork(): void {
    this.frame?.dispose();
    this.frame = null;
    this.network?.dispose();
    this.network = null;
    this.historyValid = false;
    this.frameIndex = 0;
  }

  private async rebuild(): Promise<void> {
    const generation = ++this.generation;
    // Wait for an in-flight frame: it may be using the network about to be disposed.
    while (this.busy) await new Promise((done) => setTimeout(done, 5));
    this.releaseNetwork();
    const builder = this.builder;
    if (!builder) {
      this.setState('none', 'NR off: no network');
      return;
    }
    const { width, height } = this.size;
    this.setState('building', `building the network for ${width}x${height}`);
    let network: DlssNrNetwork | null = null;
    let frame: TslFrame | null = null;
    try {
      network = await builder({
        renderer: this.renderer,
        width,
        height,
        onProgress: (message) => {
          if (generation === this.generation) this.setState('building', message);
        },
      });
      if (!isExternalFrame(network)) {
        frame = createTslFrame(network, this.renderTarget, {
          rejectOffscreenHistory: this.rejectOffscreenHistory,
          blendScale: backendBlendScale(network, this.fallbackBlendScale),
        });
        // Compile the frame kernels now rather than on the first frame.
        await this.renderer.compileComputeAsync([
          frame.kernels.inputFeatures[0].node,
          frame.kernels.inputFeatures[1].node,
          frame.kernels.compose[0].node,
          frame.kernels.compose[1].node,
          frame.unpack,
        ]);
      }
    } catch (error) {
      frame?.dispose();
      network?.dispose();
      if (generation === this.generation) {
        this.setState('failed', error instanceof Error ? error.message : String(error));
        throw error;
      }
      return;
    }
    if (generation !== this.generation) {
      // Superseded (another setNetwork / setSize / dispose) while building.
      frame?.dispose();
      network.dispose();
      return;
    }
    this.network = network;
    this.frame = frame;
    this.historyValid = false;
    this.frameIndex = 0;
    const backend = isExternalFrame(network) ? network.backend : network;
    this.setState('ready', `${backend.label}: ${backend.dispatchCount} dispatches at ${width}x${height}`);
  }
}

/** The pass's frame for a bare backend: chunk D's kernels plus the image unpack. */
interface TslFrame {
  params: NRFrameParams;
  kernels: FrameKernels;
  history: NRHistory;
  image: BufferSource;
  /** The unpack compute node (image -> `output`). */
  unpack: any;
  /** rgba8unorm, display-ready sRGB bytes. */
  output: any;
  dispose(): void;
}

function createTslFrame(
  backend: NRBackend,
  target: any,
  options: { rejectOffscreenHistory: boolean; blendScale: number },
): TslFrame {
  const g = backend.geometry;
  const { validWidth: width, validHeight: height } = g;
  const params = new NRFrameParams({ blendScale: options.blendScale });
  const history = new NRHistory(width, height);
  const image: BufferSource = { attribute: wordAttribute(width * height * 4) };
  const kernels = createFrameKernels(
    {
      fullWidth: g.fullWidth,
      fullHeight: g.fullHeight,
      validWidth: width,
      validHeight: height,
      rejectOffscreenHistory: options.rejectOffscreenHistory,
    },
    {
      color: { texture: target.textures[0] },
      motion: { velocityTexture: target.textures[1] },
      features: backend.features,
      head: backend.head,
      params,
      history,
      image,
    },
  );
  const presented = new StorageTexture(width, height);
  presented.name = 'three-dlss-nr output';
  presented.generateMipmaps = false;
  const words = storage(image.attribute, 'uint', image.attribute.count).toReadOnly();
  // The reference's canvas image (bgra8 in a u32, frame.wgsl `compose`) into a texture three can sample.
  const unpack = Fn(() => {
    const x = workgroupId.x.mul(u(8)).add(localId.x).toVar();
    const y = workgroupId.y.mul(u(8)).add(localId.y).toVar();
    If(x.lessThan(u(width)).and(y.lessThan(u(height))), () => {
      const word: TSLNode = words.element(y.mul(u(width)).add(x)).toVar();
      const byte = (shift: number) => float(word.shiftRight(u(shift)).bitAnd(u(0xff))).div(float(255));
      textureStore(presented, uvec2(x, y), vec4(byte(16), byte(8), byte(0), float(1))).toStack();
    });
  })()
    .compute([Math.ceil(width / 8), Math.ceil(height / 8), 1], [8, 8, 1])
    .setName('nr_present_unpack');
  return {
    params,
    kernels,
    history,
    image,
    unpack,
    output: presented,
    dispose() {
      presented.dispose();
      for (const node of [...kernels.inputFeatures.map((k) => k.node), ...kernels.compose.map((k) => k.node), unpack]) {
        node.dispose?.();
      }
    },
  };
}

/** Clamp a valid size to `DLSS_NR_MAX_PIXELS` (aspect kept) and to at least 64x64, whole pixels. */
export function clampSize(width: number, height: number): { width: number; height: number } {
  if (!(width > 0 && height > 0)) throw new RangeError(`invalid size ${width}x${height}`);
  const scale = Math.min(1, Math.sqrt(DLSS_NR_MAX_PIXELS / (width * height)));
  return {
    width: Math.max(64, Math.floor(width * scale)),
    height: Math.max(64, Math.floor(height * scale)),
  };
}
