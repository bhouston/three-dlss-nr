// The network's input features from a rendered three.js frame, built on the CPU.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The lanes follow the reference's
// ports/browser-webgpu/shaders/frame.wgsl `input_features` and docs/frame.md: lanes 4-6 are the centred display
// proxy of the rendered linear colour (`proxy_component`: paper-white relative, a soft shoulder above 0.75, sRGB
// encoded, on the half grid), mirrored outside the valid image; lanes 7-9 the history, here the same frame (a first
// frame has no motion); the noise lanes and the conditioning lanes come from `syntheticFeatures` (an integer hash, no
// transcendentals). The features are built once and every renderer is given the same `Float32Array`, so the
// comparison starts after the only hardware-dependent step, as the reference's own fixtures do.

import { centredProxy, syntheticFeatures } from 'three-dlss-nr/synthetic';

import type { FieldGeometry } from './visualize.js';

const f32 = Math.fround;

/** The sRGB encoding of a linear value, clamped to [0, 1] (frame.wgsl `srgb_encode`). */
function srgbEncode(value: number): number {
  const bounded = Math.min(Math.max(value, 0), 1);
  return bounded <= f32(0.0031308)
    ? f32(f32(12.92) * bounded)
    : f32(f32(1.055) * f32(bounded ** f32(1 / 2.4)) - f32(0.055));
}

/** frame.wgsl `proxy_component` with paper white 1: the display proxy code of one linear component. */
export function proxyComponent(value: number, paperWhite = 1): number {
  const finite = Number.isFinite(value) && Math.abs(value) <= 65504 ? Math.max(value, 0) : 0;
  const relative = f32(finite / Math.max(paperWhite, f32(0.05)));
  const shaped =
    relative > f32(0.75)
      ? f32(f32(0.75) + f32(f32(0.25) * f32(1 - f32(Math.exp(f32(f32(-5.77078) * f32(relative - f32(0.75))))))))
      : relative;
  return Math.fround(srgbEncode(shaped));
}

/** preprocess.wgsl's mirror at the valid edge (`2 * valid - x - 2`), clamped into the image. */
const mirror = (coordinate: number, valid: number): number =>
  Math.min(Math.max(coordinate < valid ? coordinate : 2 * valid - coordinate - 2, 0), valid - 1);

/**
 * Features `[fullHeight * fullWidth][16]` of one rendered frame. `linear` is RGBA, `validWidth * validHeight` pixels,
 * rows top to bottom (a WebGPU render target read back as is).
 */
export function featuresFromFrame(linear: Float32Array, geometry: FieldGeometry, seed = 1): Float32Array {
  const { validWidth, validHeight, fullWidth, fullHeight } = geometry;
  if (linear.length !== validWidth * validHeight * 4) throw new RangeError('featuresFromFrame: wrong image size');
  const features = syntheticFeatures(geometry, { seed });
  const centred = new Float32Array(validWidth * validHeight * 3);
  for (let i = 0; i < validWidth * validHeight; ++i) {
    for (let c = 0; c < 3; ++c) centred[i * 3 + c] = centredProxy(proxyComponent(linear[i * 4 + c]));
  }
  for (let y = 0; y < fullHeight; ++y) {
    const sy = mirror(y, validHeight);
    for (let x = 0; x < fullWidth; ++x) {
      const source = (sy * validWidth + mirror(x, validWidth)) * 3;
      const base = (y * fullWidth + x) * 16;
      for (let c = 0; c < 3; ++c) {
        features[base + 4 + c] = centred[source + c];
        features[base + 7 + c] = centred[source + c];
      }
    }
  }
  return features;
}
