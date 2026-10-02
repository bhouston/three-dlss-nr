// Input features from a recorded display proxy, for parity harnesses (fixtures that carry a proxy, not features).
//
// Port of OpenDLSS-NR ports/browser-webgpu/shaders/preprocess.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/preprocess.wgsl) to three.js
// TSL; the reference runs it from `Network.featuresFromProxy`. The proxy is four f32 code values per source pixel; the
// sixteen lanes are the same as the frame's (frame/inputFeatures.ts): Box-Muller noise (lanes 0-2, the hardware's
// transcendentals), 1, the centred proxy (4-6) and again as the "history" (7-9), the conditioning lanes, 0.
//
// Parameters are baked (one dispatch per harness run). Deviation: where the field is more than twice the valid size,
// the mirrored source coordinate is clamped into the image (the reference reads out of bounds there).

import { If, floatBitsToUint, localId, workgroupId } from 'three/tsl';

import { conditioningLanes, mirroredSource, nrCentre, nrGaussian3 } from '../frame/proxy.js';
import { f32Bits } from '../numerics/oracle.js';
import { kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { f, fBits, loadF32, u, type TSLNode } from '../tsl/packed.js';
import type { NRKernel, NRTensor } from '../types.js';

/** The reference's `PreprocessParams` (network.js `featuresFromProxy` fills them from a fixture manifest). */
export interface PreprocessSpec {
  /** The padded field (`geometry.fullWidth` / `fullHeight`). */
  fullWidth: number;
  fullHeight: number;
  /** The valid image inside the field. */
  validWidth: number;
  validHeight: number;
  /** The proxy's own size (nearest-sampled onto the valid image). */
  sourceWidth: number;
  sourceHeight: number;
  /** Noise seed (`manifest.seed ?? 0`). */
  seed: number;
  /** Auto-mask lane pair (`manifest.autoMask`); default false. */
  autoMask?: boolean;
  /** Conditioning (`manifest.conditioning`); defaults 1, 1, -1, 0 as the reference's. */
  localTone?: number;
  localStructure?: number;
  skinStructure?: number;
  style?: number;
  /** Dispatch label; default `'preprocess'`. */
  label?: string;
}

export interface PreprocessBuffers {
  /** f32 RGBA code values, `sourceWidth * sourceHeight * 4` (bit patterns in u32 words). */
  proxy: BufferSource;
  /** f32 features `[fullWidth * fullHeight][16]`. */
  features: NRTensor;
}

/** An f32 constant from a JS number, rounded to f32 as the reference's f32 uniform holds it (signed zero kept). */
const f32 = (value: number): TSLNode => fBits(f32Bits(value));

/** `preprocess` (preprocess.wgsl): the sixteen feature lanes of every padded pixel from a display proxy. */
export function createPreprocess(spec: PreprocessSpec, buffers: PreprocessBuffers): NRKernel {
  const { fullWidth, fullHeight, validWidth, validHeight, sourceWidth, sourceHeight, seed } = spec;
  const label = spec.label ?? 'preprocess';
  const { proxy, features } = buffers;
  if (features.format !== 'f32' || features.channels !== 16 || features.allocRows < fullWidth * fullHeight) {
    throw new Error(`${label}: features must be f32 [${fullWidth * fullHeight}][16]`);
  }
  if (proxy.attribute.count < sourceWidth * sourceHeight * 4) {
    throw new Error(
      `${label}: the proxy holds ${proxy.attribute.count} words, expected ${sourceWidth * sourceHeight * 4}`,
    );
  }
  if (validWidth > fullWidth || validHeight > fullHeight) throw new Error(`${label}: valid exceeds the field`);
  return kernel({
    label,
    kind: 'preprocess',
    workgroupSize: [8, 8],
    dispatch: [Math.ceil(fullWidth / 8), Math.ceil(fullHeight / 8)],
    inputs: { proxy },
    outputs: { features },
    body: ({ proxy: source, features: out }) => {
      const x = workgroupId.x.mul(u(8)).add(localId.x).toVar();
      const y = workgroupId.y.mul(u(8)).add(localId.y).toVar();
      If(x.lessThan(u(fullWidth)).and(y.lessThan(u(fullHeight))), () => {
        const sourceX = mirroredSource(x, validWidth);
        const sourceY = mirroredSource(y, validHeight);
        const imageX = u(2)
          .mul(sourceX)
          .add(u(1))
          .mul(u(sourceWidth))
          .div(u(2 * validWidth));
        const imageY = u(2)
          .mul(sourceY)
          .add(u(1))
          .mul(u(sourceHeight))
          .div(u(2 * validHeight));
        const texel = imageY.mul(u(sourceWidth)).add(imageX).mul(u(4)).toVar();
        const rgb = [0, 1, 2].map((c) => nrCentre(loadF32(source, texel.add(u(c)))).toVar());
        const noise = nrGaussian3(x, y, u(seed >>> 0)).toVar();
        const lanes = [
          noise.x,
          noise.y,
          noise.z,
          f(1),
          ...rgb,
          ...rgb,
          ...conditioningLanes({
            style: f32(spec.style ?? 0),
            localTone: f32(spec.localTone ?? 1),
            localStructure: f32(spec.localStructure ?? 1),
            skinStructure: f32(spec.skinStructure ?? -1),
            autoMask: f(spec.autoMask ? 1 : -1),
          }),
          f(0),
        ];
        const base = y.mul(u(fullWidth)).add(x).mul(u(16)).toVar();
        lanes.forEach((value, lane) => out.element(base.add(u(lane))).assign(floatBitsToUint(value)));
      });
    },
  });
}
