// The reference's own frame around the reference network, fed from a three.js render target with no CPU readback.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The frame kernels (`input_features`, `compose`
// in shaders/frame.wgsl), their parameter block, the history ping-pong and its copy are the upstream demo's
// (ports/browser-webgpu/demo/src/backend/production-pipeline.js), recorded into the graph exactly as it records
// them. What differs is how a frame arrives. The upstream demo renders with WebGL, reads the frame back to the CPU
// (RGBA16F colour, RG16F uv velocity) and uploads it with `flip_y = 1`. Here the frame is already on the same
// GPUDevice, in a three.js render target, so one small compute pass (PACK_WGSL below, written for three-dlss-nr)
// copies it into the reference's `scene` and `motion` buffers in the layout the upstream expects, with
// `flip_y = 0`:
//   * scene: the colour texel's four halves, two per word;
//   * motion: three's `velocity` node is current NDC minus previous NDC with y up; the reference wants the uv offset
//     from the current pixel to its previous position with y down: (-v.x / 2, v.y / 2), exact in f32.
// Halves are published with the upstream's round-to-nearest-even `f16_bits` (numerics.wgsl is prepended), not
// `pack2x16float`, whose rounding WGSL leaves to the implementation. On D3D12 `pack2x16float` truncates, and halving
// a subnormal half with an odd mantissa is a tie; the Chrome parity check against the demo's CPU conversion found it.
// After `compose`, a second small pass (PRESENT_WGSL) unpacks the upstream's canvas-ready bgra8 image into a three
// `StorageTexture` (rgba8unorm) the app can draw. History semantics are the upstream's: `historyValid` is false on
// the first frame, after a resize (rebuild) and after `resetHistory()`; the frame index (the noise seed) counts
// rendered frames.

import { StorageTexture } from 'three/webgpu';

import type { NRFrameTiming } from '../backend/NRBackend.js';
import { BUFFER_USAGE, TEXTURE_USAGE } from '../backend/gpuFlags.js';
import { NRFrameTimer } from '../backend/timing.js';
import { align, SHADERS } from '../../vendor/opendlss-nr/index.js';
import { ReferenceWgslBackend, type ReferenceWgslCreateOptions } from './ReferenceWgslBackend.js';

/** The upstream demo's NR controls (`demo/src/backend/nr-settings.js` `NR_DEFAULTS`). */
export interface ReferenceFrameSettings {
  enabled?: boolean;
  intensity?: number;
  localTone?: number;
  localStructure?: number;
  skinStructure?: number;
  autoMask?: boolean;
  style?: number;
}

const NR_DEFAULTS: Required<ReferenceFrameSettings> = {
  enabled: true,
  intensity: 1,
  localTone: 1,
  localStructure: 1,
  skinStructure: -1,
  autoMask: true,
  style: 0,
};

export interface ReferenceFrameOptions extends Omit<ReferenceWgslCreateOptions, 'extraShaders' | 'before' | 'after'> {
  /** Scene value that maps to display white (upstream `paperWhite`, default 1). */
  paperWhite?: number;
  /** How much of the network's colour change to keep (upstream `colorStrength`, default 1). */
  colorStrength?: number;
}

/** A three.js texture (e.g. `renderTarget.textures[i]`) or a raw GPUTexture on the renderer's device. */
export type FrameTexture = GPUTexture | { isTexture: true };

export interface ReferenceFrameInput {
  /** Linear HDR colour, RGBA16F (HalfFloatType), `width x height`, top row first (three's WebGPU convention). */
  color: FrameTexture;
  /** three's `velocity` MRT output (current NDC - previous NDC, y up), same size; omit for zero motion. */
  velocity?: FrameTexture;
  /** Drop the history before this frame (camera cut). */
  reset?: boolean;
  settings?: ReferenceFrameSettings;
  /** Measure GPU time with timestamp queries (default false). */
  timing?: boolean;
}

const PARAM_WORDS = 18;

/** scene/motion from three's textures, in the layout production-pipeline.js uploads (here with flip_y = 0). */
const PACK_WGSL = /* wgsl */ `
struct PackParams { width : u32, height : u32, has_velocity : u32, pad : u32 };
@group(0) @binding(0) var<uniform> params : PackParams;
@group(0) @binding(1) var color : texture_2d<f32>;
@group(0) @binding(2) var velocity : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> scene : array<u32>;
@group(0) @binding(4) var<storage, read_write> motion : array<u32>;

@compute @workgroup_size(8, 8)
fn pack(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  let pixel = id.y * params.width + id.x;
  let c = textureLoad(color, vec2<i32>(id.xy), 0);
  scene[pixel * 2u] = f16_bits(c.r) | (f16_bits(c.g) << 16u);
  scene[pixel * 2u + 1u] = f16_bits(c.b) | (f16_bits(c.a) << 16u);
  var uv = vec2<f32>(0.0);
  if (params.has_velocity != 0u) {
    let v = textureLoad(velocity, vec2<i32>(id.xy), 0).xy;
    uv = vec2<f32>(-v.x * 0.5, v.y * 0.5);
  }
  motion[pixel] = f16_bits(uv.x) | (f16_bits(uv.y) << 16u);
}
`;

/** The upstream's bgra8 canvas image (rows padded to 256 bytes) into an rgba8unorm storage texture. */
const PRESENT_WGSL = /* wgsl */ `
struct PresentParams { width : u32, height : u32, pitch : u32, pad : u32 };
@group(0) @binding(0) var<uniform> params : PresentParams;
@group(0) @binding(1) var<storage, read> image : array<u32>;
@group(0) @binding(2) var presented : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn present(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  let word = image[id.y * params.pitch + id.x];
  let bgra = vec4<f32>(vec4<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu, word >> 24u)) / 255.0;
  textureStore(presented, vec2<i32>(id.xy), bgra.zyxw);
}
`;

const isGpuTexture = (texture: FrameTexture): texture is GPUTexture =>
  typeof (texture as GPUTexture).createView === 'function';

/**
 * The reference network plus the reference frame (input features, compose, history), driven from three.js
 * textures. `output` holds the display-ready image (the upstream's ACES fit + sRGB display transform already
 * applied): draw it with no further tone mapping or colour conversion.
 */
export class ReferenceFrame {
  readonly backend: ReferenceWgslBackend;
  /** The presented image: rgba8unorm, sRGB-encoded bytes, `width x height`. */
  readonly output: any;
  readonly width: number;
  readonly height: number;
  paperWhite: number;
  colorStrength: number;
  private readonly device: GPUDevice;
  private readonly renderer: any;
  private readonly resources: {
    scene: GPUBuffer;
    motion: GPUBuffer;
    history: [GPUBuffer, GPUBuffer];
    image: GPUBuffer;
    featuresPass: { index: number };
    composePass: { index: number };
    packParams: GPUBuffer;
    presentParams: GPUBuffer;
    zeroVelocity: GPUTexture;
  };
  private readonly packPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPUComputePipeline;
  private readonly presentGroup: GPUBindGroup;
  private packGroup: { color: GPUTexture; velocity: GPUTexture; group: GPUBindGroup } | null = null;
  private readonly timer: NRFrameTimer;
  private historyValid = false;
  private validated = false;
  private frameIndex = 0;

  private constructor(fields: {
    backend: ReferenceWgslBackend;
    renderer: any;
    output: any;
    resources: ReferenceFrame['resources'];
    packPipeline: GPUComputePipeline;
    presentPipeline: GPUComputePipeline;
    presentGroup: GPUBindGroup;
    paperWhite: number;
    colorStrength: number;
  }) {
    this.backend = fields.backend;
    this.renderer = fields.renderer;
    this.device = fields.backend.device;
    this.output = fields.output;
    this.resources = fields.resources;
    this.packPipeline = fields.packPipeline;
    this.presentPipeline = fields.presentPipeline;
    this.presentGroup = fields.presentGroup;
    this.paperWhite = fields.paperWhite;
    this.colorStrength = fields.colorStrength;
    this.width = fields.backend.geometry.validWidth;
    this.height = fields.backend.geometry.validHeight;
    this.timer = new NRFrameTimer(this.device);
  }

  static async create(options: ReferenceFrameOptions): Promise<ReferenceFrame> {
    const { renderer, width, height } = options;
    const paperWhite = options.paperWhite ?? 1;
    const colorStrength = options.colorStrength ?? 1;
    let resources: ReferenceFrame['resources'] | undefined;
    let device!: GPUDevice;
    // The same parameter block the upstream writes before recording; rewritten every frame by `render`.
    const initialParams = (net: any) => frameParams(net.geometry, NR_DEFAULTS, false, 0, paperWhite, colorStrength, 1);
    const backend = await ReferenceWgslBackend.create({
      ...options,
      extraShaders: [
        { name: 'shaders/frame.wgsl', code: SHADERS['frame.wgsl'], entryPoints: ['input_features', 'compose'] },
      ],
      // production-pipeline.js `build`, `before` and `after`, minus the staging slots (the frame is on the GPU).
      before: (net) => {
        device = net.device;
        const storage = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;
        const scene = device.createBuffer({ label: 'rendered frame', size: width * height * 8, usage: storage });
        const motion = device.createBuffer({ label: 'velocity', size: width * height * 4, usage: storage });
        const history: [GPUBuffer, GPUBuffer] = [0, 1].map((index) =>
          device.createBuffer({ label: `history ${index}`, size: width * height * 8, usage: storage }),
        ) as [GPUBuffer, GPUBuffer];
        const image = device.createBuffer({
          label: 'presented image',
          size: align(width * 4, 256) * height,
          usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
        });
        const g = net.geometry;
        net.recorder.pass(
          'input_features',
          { 1: scene, 2: history[0], 4: motion, 5: net.features.buffer },
          initialParams(net),
          [Math.ceil(g.fullWidth / 8), Math.ceil(g.fullHeight / 8)],
          'input features',
        );
        const featuresPass = net.recorder.passes.at(-1);
        resources = { scene, motion, history, image, featuresPass } as ReferenceFrame['resources'];
      },
      after: (net) => {
        const r = resources!;
        net.recorder.pass(
          'compose',
          { 1: r.scene, 2: r.history[0], 3: net.graph.head.buffer, 4: r.motion, 6: r.history[1], 7: r.image },
          initialParams(net),
          [Math.ceil(width / 8), Math.ceil(height / 8)],
          'compose',
        );
        r.composePass = net.recorder.passes.at(-1);
        // The graph is recorded once, so the history buffers cannot alternate by frame parity; one is read,
        // the other written, and a copy swaps them.
        net.recorder.copy(r.history[1], r.history[0], width * height * 8, 'history swap');
      },
    });
    const r = resources!;
    r.packParams = device.createBuffer({
      label: 'pack parameters',
      size: 16,
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    });
    r.presentParams = device.createBuffer({
      label: 'present parameters',
      size: 16,
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    });
    device.queue.writeBuffer(r.presentParams, 0, new Uint32Array([width, height, align(width * 4, 256) / 4, 0]));
    r.zeroVelocity = device.createTexture({
      label: 'zero velocity',
      size: [1, 1],
      format: 'rgba16float',
      usage: TEXTURE_USAGE.TEXTURE_BINDING,
    });

    const packPipeline = await device.createComputePipelineAsync({
      label: 'reference frame pack',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          label: 'reference frame pack',
          code: `${SHADERS['numerics.wgsl']}\n${PACK_WGSL}`,
        }),
        entryPoint: 'pack',
      },
    });
    const presentPipeline = await device.createComputePipelineAsync({
      label: 'reference frame present',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'reference frame present', code: PRESENT_WGSL }),
        entryPoint: 'present',
      },
    });

    const output = new StorageTexture(width, height);
    output.name = 'reference NR output';
    output.generateMipmaps = false;
    renderer.initTexture(output);
    const outputTexture: GPUTexture | undefined = renderer.backend.get(output).texture;
    if (!outputTexture || outputTexture.format !== 'rgba8unorm') {
      throw new Error(`the output StorageTexture is ${outputTexture?.format ?? 'missing'}, expected rgba8unorm`);
    }
    const presentGroup = device.createBindGroup({
      label: 'reference frame present',
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.presentParams } },
        { binding: 1, resource: { buffer: r.image } },
        { binding: 2, resource: outputTexture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
      ],
    });
    return new ReferenceFrame({
      backend,
      renderer,
      output,
      resources: r,
      packPipeline,
      presentPipeline,
      presentGroup,
      paperWhite,
      colorStrength,
    });
  }

  /** Start the next frame without history (as the upstream `resetHistory`). */
  resetHistory(): void {
    this.historyValid = false;
  }

  /**
   * One frame: pack the textures, input features, the network, compose, history swap, present. Call after the
   * scene was rendered into `input.color` (same queue, so the render is complete before the pack reads it).
   */
  async render(input: ReferenceFrameInput): Promise<NRFrameTiming> {
    const { device, resources: r, backend } = this;
    const settings = { ...NR_DEFAULTS, ...input.settings };
    const historyValid = this.historyValid && !input.reset;
    const words = frameParams(
      backend.network.geometry,
      settings,
      historyValid,
      this.frameIndex,
      this.paperWhite,
      this.colorStrength,
      backend.blendScale,
    );
    const paramsBuffer: GPUBuffer = backend.network.recorder.paramsBuffer;
    device.queue.writeBuffer(paramsBuffer, r.featuresPass.index * 256, words);
    device.queue.writeBuffer(paramsBuffer, r.composePass.index * 256, words);
    const group = this.bindPack(input);
    device.queue.writeBuffer(r.packParams, 0, new Uint32Array([this.width, this.height, input.velocity ? 1 : 0, 0]));

    const encodePack = (encoder: GPUCommandEncoder) => {
      const pass = encoder.beginComputePass({ label: 'reference frame pack' });
      pass.setPipeline(this.packPipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
      pass.end();
    };
    const encodePresent = (encoder: GPUCommandEncoder) => {
      const pass = encoder.beginComputePass({ label: 'reference frame present' });
      pass.setPipeline(this.presentPipeline);
      pass.setBindGroup(0, this.presentGroup);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
      pass.end();
    };

    // One command buffer per frame: pack, the recorded graph (input features ... compose, history swap), present.
    // The first one is checked inside a validation error scope, as the upstream  checks its first
    // frame: WebGPU drops a rejected command buffer whole, which would otherwise read as a frame of stale outputs.
    const validating = !this.validated;
    if (validating) device.pushErrorScope('validation');
    const timing = await this.timer.measure(
      () => {
        const encoder = device.createCommandEncoder({ label: 'reference nr frame' });
        encodePack(encoder);
        backend.network.recorder.encode(encoder);
        encodePresent(encoder);
        device.queue.submit([encoder.finish()]);
      },
      { gpu: !validating && (input.timing ?? false) },
    );
    if (validating) {
      const error = await device.popErrorScope();
      if (error) throw new Error();
      this.validated = true;
    }
    this.historyValid = true;
    this.frameIndex += 1;
    return timing;
  }

  /** The presented image as tightly packed RGBA8 rows (top row first). */
  async readImage(): Promise<Uint8Array> {
    const pitch = align(this.width * 4, 256);
    const raw = await this.backend.readBuffer(this.resources.image, pitch * this.height);
    const rgba = new Uint8Array(this.width * this.height * 4);
    for (let y = 0; y < this.height; ++y) {
      for (let x = 0; x < this.width; ++x) {
        const from = y * pitch + x * 4;
        const to = (y * this.width + x) * 4;
        rgba[to] = raw[from + 2];
        rgba[to + 1] = raw[from + 1];
        rgba[to + 2] = raw[from];
        rgba[to + 3] = raw[from + 3];
      }
    }
    return rgba;
  }

  /** The history the next frame reads (RGBA16F halves of the blended proxy code values), raw bytes. */
  readHistory(): Promise<Uint8Array> {
    return this.backend.readBuffer(this.resources.history[0], this.width * this.height * 8);
  }

  /** The packed `scene` and `motion` buffers of the last frame (what the upstream demo would have uploaded). */
  async readPacked(): Promise<{ scene: Uint8Array; motion: Uint8Array }> {
    return {
      scene: await this.backend.readBuffer(this.resources.scene, this.width * this.height * 8),
      motion: await this.backend.readBuffer(this.resources.motion, this.width * this.height * 4),
    };
  }

  dispose(): void {
    const r = this.resources;
    for (const buffer of [r.scene, r.motion, ...r.history, r.image, r.packParams, r.presentParams]) buffer.destroy();
    r.zeroVelocity.destroy();
    this.output.dispose();
    this.timer.dispose();
    this.backend.dispose();
  }

  private gpuTexture(texture: FrameTexture, what: string): GPUTexture {
    if (isGpuTexture(texture)) return texture;
    const gpu: GPUTexture | undefined = this.renderer.backend.get(texture).texture;
    if (!gpu) throw new Error(`the ${what} texture has no GPU texture yet; render into its target first`);
    if (gpu.width !== this.width || gpu.height !== this.height) {
      throw new Error(`the ${what} texture is ${gpu.width}x${gpu.height}; this frame is ${this.width}x${this.height}`);
    }
    if (gpu.sampleCount !== 1) throw new Error(`the ${what} texture is multisampled; render with samples: 0`);
    return gpu;
  }

  private bindPack(input: ReferenceFrameInput): GPUBindGroup {
    const color = this.gpuTexture(input.color, 'colour');
    const velocity = input.velocity ? this.gpuTexture(input.velocity, 'velocity') : this.resources.zeroVelocity;
    if (this.packGroup?.color === color && this.packGroup.velocity === velocity) return this.packGroup.group;
    const group = this.device.createBindGroup({
      label: 'reference frame pack',
      layout: this.packPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.resources.packParams } },
        { binding: 1, resource: color.createView() },
        { binding: 2, resource: velocity.createView() },
        { binding: 3, resource: { buffer: this.resources.scene } },
        { binding: 4, resource: { buffer: this.resources.motion } },
      ],
    });
    this.packGroup = { color, velocity, group };
    return group;
  }
}

/** production-pipeline.js `params`, with `flip_y = 0` (the frame comes from a WebGPU render target, top row first). */
export function frameParams(
  geometry: { fullWidth: number; fullHeight: number; validWidth: number; validHeight: number },
  settings: Required<ReferenceFrameSettings>,
  historyValid: boolean,
  frameIndex: number,
  paperWhite: number,
  colorStrength: number,
  blendScale: number,
): Uint32Array {
  const buffer = new ArrayBuffer(PARAM_WORDS * 4);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  words[0] = geometry.fullWidth;
  words[1] = geometry.fullHeight;
  words[2] = geometry.validWidth;
  words[3] = geometry.validHeight;
  words[4] = frameIndex & 0xffff;
  words[5] = historyValid ? 1 : 0;
  words[6] = settings.enabled === false ? 0 : 1;
  words[7] = align(geometry.validWidth * 4, 256) / 4;
  floats[8] = paperWhite;
  floats[9] = settings.style ?? 0;
  floats[10] = settings.localTone ?? 1;
  floats[11] = settings.localStructure ?? 1;
  floats[12] = settings.skinStructure ?? -1;
  floats[13] = settings.autoMask ? 1 : 0;
  floats[14] = blendScale;
  floats[15] = settings.intensity ?? 1;
  floats[16] = colorStrength;
  words[17] = 0;
  return words;
}
