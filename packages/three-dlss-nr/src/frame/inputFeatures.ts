// The network's sixteen input lanes from a rendered frame and the previous output.
//
// Port of `input_features` of OpenDLSS-NR ports/browser-webgpu/shaders/frame.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/frame.wgsl) to three.js TSL.
// Per padded pixel: three noise lanes, 1, the centred display proxy of the (mirrored) source pixel, the centred
// reprojected history (or the proxy again when there is none), the conditioning lanes and 0. See docs/frame.md.
//
// Differences from the reference, all outside what the reference defines: the colour and motion can be textures
// (design 4.4); the mirrored source coordinate is clamped into the image (the reference reads out of bounds when the
// field is more than twice the valid size); `rejectOffscreenHistory` adds the native pipeline's per-pixel history flag.

import { If, float, floatBitsToUint, localId, vec3, workgroupId } from 'three/tsl';

import { kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { f, u, type TSLNode } from '../tsl/packed.js';
import type { NRKernel, NRTensor } from '../types.js';
import {
  frameReaders,
  type FrameColorSource,
  type FrameGeometry,
  type FrameMotionSource,
  type NRFrameParams,
} from './frameInputs.js';
import { conditioningLanes, mirroredSource, nrCentre3, nrGaussian3, nrProxyComponent } from './proxy.js';
import { historySampler } from './reproject.js';

export interface InputFeaturesSpec extends FrameGeometry {
  /** Dispatch label; default `'input features'` (the reference's). */
  label?: string;
}

export interface InputFeaturesBuffers {
  color: FrameColorSource;
  motion: FrameMotionSource;
  /** The previous frame's output: rgba16float, `validWidth * validHeight` pixels (`NRHistory`). */
  history: BufferSource;
  /** f32 `[fullWidth * fullHeight][16]` (the graph's `input features` tensor). */
  features: NRTensor;
  params: NRFrameParams;
}

/** True where the reprojected uv (pixel centre + motion) lies inside [0, 1]^2. */
export function historyOnScreen(x: TSLNode, y: TSLNode, motion: TSLNode, width: number, height: number): TSLNode {
  const ux = float(x).add(f(0.5)).div(f(width)).add(motion.x);
  const uy = float(y).add(f(0.5)).div(f(height)).add(motion.y);
  return ux
    .greaterThanEqual(f(0))
    .and(ux.lessThanEqual(f(1)))
    .and(uy.greaterThanEqual(f(0)))
    .and(uy.lessThanEqual(f(1)));
}

/** `input_features` (frame.wgsl): one invocation per padded pixel, workgroups of 8x8. */
export function createInputFeatures(spec: InputFeaturesSpec, buffers: InputFeaturesBuffers): NRKernel {
  const { fullWidth, fullHeight, validWidth, validHeight } = spec;
  const label = spec.label ?? 'input features';
  const { color, motion, history, features, params } = buffers;
  if (features.format !== 'f32' || features.channels !== 16 || features.allocRows < fullWidth * fullHeight) {
    throw new Error(`${label}: features must be f32 [${fullWidth * fullHeight}][16]`);
  }
  if (validWidth > fullWidth || validHeight > fullHeight) throw new Error(`${label}: valid exceeds the field`);
  if (history.attribute.count < validWidth * validHeight * 2) throw new Error(`${label}: history is too small`);
  const readers = frameReaders(spec, color, motion);
  return kernel({
    label,
    kind: 'input_features',
    workgroupSize: [8, 8],
    dispatch: [Math.ceil(fullWidth / 8), Math.ceil(fullHeight / 8)],
    inputs: { ...readers.inputs, history },
    outputs: { features },
    body: (views) => {
      const v = views as Record<string, TSLNode>;
      const sampler = historySampler(v.history, validWidth, validHeight);
      const x = workgroupId.x.mul(u(8)).add(localId.x).toVar();
      const y = workgroupId.y.mul(u(8)).add(localId.y).toVar();
      If(x.lessThan(u(fullWidth)).and(y.lessThan(u(fullHeight))), () => {
        const sourceX = mirroredSource(x, validWidth).toVar();
        const sourceY = mirroredSource(y, validHeight).toVar();
        const rgb = readers.sceneAt(v, sourceX, sourceY).toVar();
        const proxy = (c: TSLNode) => nrProxyComponent(c, params.paperWhite);
        const centred = nrCentre3(vec3(proxy(rgb.x), proxy(rgb.y), proxy(rgb.z))).toVar();
        const previous = vec3(centred).toVar();
        const motionUv = readers.motionAt(v, sourceX, sourceY).toVar();
        let useHistory = params.historyValid.notEqual(u(0));
        if (spec.rejectOffscreenHistory) {
          useHistory = useHistory.and(historyOnScreen(sourceX, sourceY, motionUv, validWidth, validHeight));
        }
        If(useHistory, () => {
          previous.assign(nrCentre3(sampler.reproject(sourceX, sourceY, motionUv)));
        });
        const noise = nrGaussian3(x, y, params.seed).toVar();
        const lanes = [
          noise.x,
          noise.y,
          noise.z,
          f(1),
          centred.x,
          centred.y,
          centred.z,
          previous.x,
          previous.y,
          previous.z,
          ...conditioningLanes(params),
          f(0),
        ];
        const base = y.mul(u(fullWidth)).add(x).mul(u(16)).toVar();
        lanes.forEach((value, lane) => v.features.element(base.add(u(lane))).assign(floatBitsToUint(value)));
      });
    },
  });
}
