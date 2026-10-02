// What the frame kernels read: the rendered colour and motion (from three.js textures or from storage buffers), and
// the per-frame parameters as TSL uniforms.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The reference's frame.wgsl reads a rendered frame
// that crossed from WebGL on the CPU: rgba16float colour and rg16float motion in storage buffers, optionally
// bottom-up (`flip_y`), and a `Params` uniform block. Here the same kernels can read a WebGPURenderer render directly
// (design section 4.4): the colour attachment and three's `velocity` MRT output as textures, no readback.

import { ivec2, int, textureLoad, uniform, vec2, vec3 } from 'three/tsl';

import type { BufferSource } from '../tsl/KernelBuilder.js';
import { nrF16ToF32 } from '../tsl/numerics.js';
import { f, u, type TSLNode } from '../tsl/packed.js';
import { rgba16Rgb } from './reproject.js';

/** A three.js texture (`Texture`, `RenderTarget.textures[i]`, `StorageTexture`); three ships no declarations. */
export type ThreeTexture = any;

/**
 * The rendered frame, linear HDR (the network's proxy is derived from it, and with NR off it is what is presented).
 *   * `{ texture }`: any RGBA texture three can `textureLoad` (e.g. a `HalfFloatType` render target's colour), read
 *     at integer pixel coordinates, row 0 = the top of the image (WebGPU's convention).
 *   * `{ buffer }`: rgba16float, two words per pixel, `validWidth * validHeight` pixels, top-down rows (the reference's
 *     `scene` binding).
 */
export type FrameColorSource = { texture: ThreeTexture } | { buffer: BufferSource };

/**
 * Per-pixel motion to where the surface was last frame.
 *   * `{ velocityTexture }`: three's `VelocityNode` output (`velocity` in an MRT): current NDC minus previous NDC, y up.
 *     Converted to the reference convention as `(-v.x / 2, v.y / 2)`.
 *   * `{ texture }`: motion already in the reference convention (rg = uv units, current -> previous, y down).
 *   * `{ buffer }`: rg16float in the reference convention, one word per pixel (the reference's `motion` binding).
 *   * `null`: no motion (a static camera and scene).
 */
export type FrameMotionSource =
  | { velocityTexture: ThreeTexture }
  | { texture: ThreeTexture }
  | { buffer: BufferSource }
  | null;

/** Geometry shared by the frame kernels (baked into their node graphs; a resize rebuilds them). */
export interface FrameGeometry {
  /** The padded field the network runs on (`geometry.fullWidth` / `fullHeight`). */
  fullWidth: number;
  fullHeight: number;
  /** The rendered image inside it. */
  validWidth: number;
  validHeight: number;
  /**
   * The colour and motion arrive bottom-up (a WebGL readback; frame.wgsl `flip_y`): rows are read flipped and the
   * motion's v negated. Default false; never needed for a WebGPURenderer render.
   */
  flipY?: boolean;
  /**
   * Per-pixel history rejection: a pixel whose reprojected position (uv + motion) leaves [0, 1] gets no history (lanes
   * 7-9 = the current proxy, blend weight 0), as the native pipeline's history flag does (docs/frame.md). Default
   * false, which is the reference WebGPU port's behaviour (only the global `historyValid`).
   */
  rejectOffscreenHistory?: boolean;
}

/** Per-frame settings (the reference's `Params` floats and flags; production-pipeline.js `params`). */
export interface NRFrameSettings {
  /** Noise seed; the reference uses `frameIndex & 0xffff`. */
  seed?: number;
  /** The history holds the previous frame (false after a reset or a resize). */
  historyValid?: boolean;
  /** NR on; off presents the rendered frame through the same output path. */
  enabled?: boolean;
  /** Scene value that maps to the proxy's 1.0 (before the shoulder). Default 1. */
  paperWhite?: number;
  /** 0 none, 1 cinematic, 2 natural (a grade on the network's output). Default 0. */
  style?: number;
  localTone?: number;
  localStructure?: number;
  skinStructure?: number;
  autoMask?: boolean;
  /** The model's learned cap on the history blend weight (`model.blendScale()`). */
  blendScale?: number;
  /** 1 = the full correction; lower dials it back towards the proxy. */
  intensity?: number;
  /** 0 = the network moves luminance only, 1 = colour too. */
  colorStrength?: number;
}

/** The per-frame parameters as TSL uniform nodes, shared by the input-features and compose kernels (R13). */
export class NRFrameParams {
  readonly seed = uniform(0, 'uint');
  readonly historyValid = uniform(0, 'uint');
  readonly enabled = uniform(1, 'uint');
  readonly paperWhite = uniform(1, 'float');
  readonly style = uniform(0, 'float');
  readonly localTone = uniform(1, 'float');
  readonly localStructure = uniform(1, 'float');
  readonly skinStructure = uniform(-1, 'float');
  readonly autoMask = uniform(0, 'float');
  readonly blendScale = uniform(1, 'float');
  readonly intensity = uniform(1, 'float');
  readonly colorStrength = uniform(1, 'float');

  constructor(settings: NRFrameSettings = {}) {
    this.set(settings);
  }

  /** Update the uniforms (only the fields given); takes effect on the next `renderer.compute`. */
  set(settings: NRFrameSettings): this {
    const s = settings;
    if (s.seed !== undefined) this.seed.value = s.seed >>> 0;
    if (s.historyValid !== undefined) this.historyValid.value = s.historyValid ? 1 : 0;
    if (s.enabled !== undefined) this.enabled.value = s.enabled ? 1 : 0;
    if (s.paperWhite !== undefined) this.paperWhite.value = s.paperWhite;
    if (s.style !== undefined) this.style.value = s.style;
    if (s.localTone !== undefined) this.localTone.value = s.localTone;
    if (s.localStructure !== undefined) this.localStructure.value = s.localStructure;
    if (s.skinStructure !== undefined) this.skinStructure.value = s.skinStructure;
    if (s.autoMask !== undefined) this.autoMask.value = s.autoMask ? 1 : 0;
    if (s.blendScale !== undefined) this.blendScale.value = s.blendScale;
    if (s.intensity !== undefined) this.intensity.value = s.intensity;
    if (s.colorStrength !== undefined) this.colorStrength.value = s.colorStrength;
    return this;
  }
}

/** Readers of the colour and motion sources inside one kernel. */
export interface FrameReaders {
  /** Storage buffers the kernel must bind (buffer sources), under the names the readers use. */
  inputs: Record<string, BufferSource>;
  /** Linear RGB of the rendered frame at top-down pixel (x, y) (`scene_at`). */
  sceneAt(views: Record<string, TSLNode>, x: TSLNode, y: TSLNode): TSLNode;
  /** Motion at top-down pixel (x, y) in uv units, current -> previous, y down (`motion_at`). */
  motionAt(views: Record<string, TSLNode>, x: TSLNode, y: TSLNode): TSLNode;
}

/** Build the readers of a kernel's colour and motion sources. */
export function frameReaders(
  geometry: FrameGeometry,
  color: FrameColorSource,
  motion: FrameMotionSource,
): FrameReaders {
  const { validWidth, validHeight } = geometry;
  const flip = geometry.flipY === true;
  const row = (y: TSLNode): TSLNode => (flip ? u(validHeight - 1).sub(y) : y);
  const inputs: Record<string, BufferSource> = {};
  if ('buffer' in color) inputs.scene = color.buffer;
  if (motion && 'buffer' in motion) inputs.motion = motion.buffer;
  const load = (texture: ThreeTexture, x: TSLNode, y: TSLNode): TSLNode =>
    textureLoad(texture, ivec2(int(x), int(row(y))));

  const sceneAt = (views: Record<string, TSLNode>, x: TSLNode, y: TSLNode): TSLNode =>
    'buffer' in color ? rgba16Rgb(views.scene, row(y).mul(u(validWidth)).add(x)) : vec3(load(color.texture, x, y).xyz);

  const motionAt = (views: Record<string, TSLNode>, x: TSLNode, y: TSLNode): TSLNode => {
    let value: TSLNode;
    if (motion === null) return vec2(f(0), f(0));
    if ('buffer' in motion) {
      const word = views.motion.element(row(y).mul(u(validWidth)).add(x)).toVar();
      value = vec2(nrF16ToF32(word.bitAnd(u(0xffff))), nrF16ToF32(word.shiftRight(u(16))));
    } else if ('velocityTexture' in motion) {
      const v = load(motion.velocityTexture, x, y);
      value = vec2(v.x.mul(f(-0.5)), v.y.mul(f(0.5)));
    } else {
      value = vec2(load(motion.texture, x, y).xy);
    }
    return flip ? vec2(value.x, value.y.negate()) : value;
  };

  return { inputs, sceneAt, motionAt };
}
