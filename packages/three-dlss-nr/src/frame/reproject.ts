// Where a pixel was last frame: the history read at the motion-reprojected position with a five-tap Catmull-Rom.
//
// Port of `history_texel`, `history_bilinear` and `reprojected_history` of OpenDLSS-NR
// ports/browser-webgpu/shaders/frame.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/frame.wgsl) to three.js TSL.
// The filter is not decoration: the history is fed back into the network's own input, so a softer fetch compounds
// frame over frame (docs/frame.md, "History reconstruction").

import { Fn, clamp, float, floor, int, mix, uint, vec2, vec3 } from 'three/tsl';

import { nrF16ToF32 } from '../tsl/numerics.js';
import { f, i, u, type TSLNode } from '../tsl/packed.js';

/** RGB of pixel `pixel` of an rgba16float buffer (two words per pixel, `scene_rgb`). */
export function rgba16Rgb(buffer: TSLNode, pixel: TSLNode): TSLNode {
  const base = pixel.mul(u(2)).toVar();
  const rg = buffer.element(base).toVar();
  const ba = buffer.element(base.add(u(1)));
  return vec3(nrF16ToF32(rg.bitAnd(u(0xffff))), nrF16ToF32(rg.shiftRight(u(16))), nrF16ToF32(ba.bitAnd(u(0xffff))));
}

/**
 * The history sampler of one kernel over `history` (an rgba16float storage view of `width x height` pixels). Returns
 * `reproject(px, py, motion)`: the history at pixel (px, py) moved by `motion` (uv units, current -> previous, y down).
 */
export function historySampler(history: TSLNode, width: number, height: number) {
  const texel = (x: TSLNode, y: TSLNode): TSLNode => {
    const cx = uint(clamp(x, i(0), i(width - 1)));
    const cy = uint(clamp(y, i(0), i(height - 1)));
    return rgba16Rgb(history, cy.mul(u(width)).add(cx));
  };

  const bilinear = Fn(([uv]: [TSLNode]) => {
    const size = vec2(f(width), f(height));
    const position = uv
      .mul(size)
      .sub(vec2(f(0.5)))
      .toVar();
    const base = floor(position).toVar();
    const t = position.sub(base).toVar();
    const x = int(base.x).toVar();
    const y = int(base.y).toVar();
    const top = mix(texel(x, y), texel(x.add(i(1)), y), t.x);
    const bottom = mix(texel(x, y.add(i(1))), texel(x.add(i(1)), y.add(i(1))), t.x);
    return mix(top, bottom, t.y);
  }).setLayout({ name: 'nr_history_bilinear', type: 'vec3', inputs: [{ name: 'uv', type: 'vec2' }] });

  const reproject = Fn(([px, py, motion]: [TSLNode, TSLNode, TSLNode]) => {
    const valid = vec2(f(width), f(height));
    const half = vec2(f(0.5));
    const one = vec2(f(1));
    const uv = vec2(float(px), float(py)).add(half).div(valid);
    const position = motion.add(uv).mul(valid).toVar();
    const base = floor(position.sub(half)).add(half).toVar();
    const t = clamp(position.sub(base), vec2(f(0)), one).toVar();
    const square = t.mul(t).toVar();
    const cube = t.mul(square).toVar();
    const w0 = square.sub(t.add(cube).mul(f(0.5))).toVar();
    const w1 = cube
      .mul(f(1.5))
      .sub(square.mul(f(2.5)))
      .add(one)
      .toVar();
    const w3 = cube.sub(square).mul(f(0.5)).toVar();
    const w2 = one.sub(w0).sub(w1).sub(w3).toVar();
    const middle = w1.add(w2).toVar();
    const lowEdge = vec2(f(0.5));
    const highEdge = valid.sub(half);
    const low = clamp(base.sub(one), lowEdge, highEdge).div(valid).toVar();
    const center = clamp(base.add(w2.div(middle)), lowEdge, highEdge)
      .div(valid)
      .toVar();
    const high = clamp(base.add(vec2(f(2))), lowEdge, highEdge)
      .div(valid)
      .toVar();
    const a = w0.x.mul(middle.y).toVar();
    const b = w0.y.mul(middle.x).toVar();
    const c = middle.x.mul(middle.y).toVar();
    const d = w3.y.mul(middle.x).toVar();
    const e = w3.x.mul(middle.y).toVar();
    const value = bilinear(vec2(low.x, center.y))
      .mul(a)
      .add(bilinear(vec2(center.x, low.y)).mul(b))
      .toVar();
    value.assign(bilinear(center).mul(c).add(value));
    value.assign(bilinear(vec2(center.x, high.y)).mul(d).add(value));
    value.assign(bilinear(vec2(high.x, center.y)).mul(e).add(value));
    return value.mul(f(1).div(e.add(d.add(c.add(a.add(b))))));
  }).setLayout({
    name: 'nr_reprojected_history',
    type: 'vec3',
    inputs: [
      { name: 'px', type: 'uint' },
      { name: 'py', type: 'uint' },
      { name: 'motion', type: 'vec2' },
    ],
  });

  return { texel, bilinear, reproject };
}
