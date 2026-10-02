// The network's head back into a picture, and the next frame's history.
//
// Port of `compose` and the display operators of OpenDLSS-NR ports/browser-webgpu/shaders/frame.wgsl (MIT, (c) 2026
// maan, https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/frame.wgsl) to three.js
// TSL. Per valid pixel: the head's RGB residual on the proxy, the learned blend with the reprojected history, the
// history stored truncated to the half grid, then the optional style grade, intensity, the tone upgrade back onto the
// HDR scene and the colour-strength blend (docs/frame.md, "Composition").
//
// Outputs (any combination):
//   * `nextHistory` (always): rgba16float, what the next frame's input features and blend read.
//   * `outputTexture`: a three.js `StorageTexture` (RGBA16F/RGBA32F) receiving the composed **linear HDR** scene (with
//     NR off: the rendered scene itself). Present it through three's `ACESFilmicToneMapping` + sRGB output, which is the
//     same operator as the reference's `display_transform`, so NR on and off are compared through one display path
//     (design 4.4).
//   * `image`: the reference's packed bgra8 canvas buffer (display transform applied here), for parity and readback.

import {
  Fn,
  If,
  abs,
  clamp,
  dot,
  exp2,
  float,
  length,
  localId,
  log2,
  max,
  min,
  pow,
  round,
  sign,
  textureStore,
  uvec2,
  uvec3,
  vec3,
  vec4,
  workgroupId,
} from 'three/tsl';

import { kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { nrF16ToF32 } from '../tsl/numerics.js';
import { f, fBits, loadF32, pick, u, type TSLNode } from '../tsl/packed.js';
import type { NRKernel, NRTensor } from '../types.js';
import {
  frameReaders,
  type FrameColorSource,
  type FrameGeometry,
  type FrameMotionSource,
  type NRFrameParams,
  type ThreeTexture,
} from './frameInputs.js';
import { historyOnScreen } from './inputFeatures.js';
import { fr, nrFma, nrProxyComponent, nrSrgbDecode, nrSrgbEncode, nrTruncateHalf } from './proxy.js';
import { historySampler } from './reproject.js';

const constant3 = (values: readonly [number, number, number]): TSLNode =>
  vec3(fr(values[0]), fr(values[1]), fr(values[2]));

/** `row3`: a 3x3 matrix given by rows, times a vector. */
const row3 = (
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
  value: TSLNode,
): TSLNode => vec3(dot(constant3(a), value), dot(constant3(b), value), dot(constant3(c), value));

const vec3Layout = (name: string, inputs: string[]) => ({
  name,
  type: 'vec3',
  inputs: inputs.map((input) => ({ name: input, type: 'vec3' })),
});

/** `luminance` (Rec. 709 weights). */
export const nrLuminance = (color: TSLNode): TSLNode => dot(color, constant3([0.212639, 0.715169, 0.072192]));

export const nrToOklab = Fn(([color]: [TSLNode]) => {
  const lms = row3(
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
    color,
  ).toVar();
  const cubeRoot = sign(lms).mul(pow(abs(lms), vec3(fr(1 / 3))));
  return row3(
    [0.2104542553, 0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766],
    cubeRoot,
  );
}).setLayout(vec3Layout('nr_to_oklab', ['color']));

export const nrFromOklab = Fn(([lab]: [TSLNode]) => {
  const lms = row3(
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.291485548],
    lab,
  ).toVar();
  return row3(
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.707614701],
    lms.mul(lms).mul(lms),
  );
}).setLayout(vec3Layout('nr_from_oklab', ['lab']));

/** `clamp_ap1`: back through AP1 with the negatives removed. */
export const nrClampAp1 = Fn(([color]: [TSLNode]) => {
  const ap1 = max(
    row3([0.613097, 0.339523, 0.047379], [0.070194, 0.916354, 0.013452], [0.020616, 0.10957, 0.869815], color),
    vec3(f(0)),
  );
  return row3(
    [1.705051, -0.621792, -0.083259],
    [-0.130256, 1.140805, -0.010548],
    [-0.024003, -0.128969, 1.152972],
    ap1,
  );
}).setLayout(vec3Layout('nr_clamp_ap1', ['color']));

/** `hue_oklab`: the lightness of one colour carrying the hue and chroma direction of another. */
export const nrHueOklab = Fn(([incorrect, correct]: [TSLNode, TSLNode]) => {
  const result = nrToOklab(incorrect).toVar();
  const correctLab = nrToOklab(correct).toVar();
  const incorrectChroma = length(result.yz);
  const correctChroma = length(correctLab.yz).toVar();
  const scale = pick(correctChroma.equal(f(0)), f(1), incorrectChroma.div(correctChroma));
  return nrClampAp1(nrFromOklab(vec3(result.x, correctLab.y.mul(scale), correctLab.z.mul(scale))));
}).setLayout(vec3Layout('nr_hue_oklab', ['incorrect', 'correct']));

/** `upgrade_tone_map`: the neural LDR result rescaled onto the HDR original's luminance, hue kept in Oklab. */
export const nrUpgradeToneMap = Fn(([original, proxy, neural]: [TSLNode, TSLNode, TSLNode]) => {
  const originalY = nrLuminance(original).toVar();
  const proxyY = nrLuminance(proxy).toVar();
  const neuralY = nrLuminance(neural).toVar();
  const result = vec3(original).toVar();
  If(neuralY.greaterThan(fr(0.00001)), () => {
    const ratio = pick(
      originalY.lessThan(proxyY),
      originalY.div(max(proxyY, fr(0.000001))),
      neuralY.add(max(f(0), originalY.sub(proxyY))).div(neuralY),
    );
    result.assign(original.add(nrHueOklab(neural.mul(ratio), neural).sub(original)));
  });
  return result;
}).setLayout(vec3Layout('nr_upgrade_tone_map', ['original', 'proxy', 'neural']));

/** `aces_fit`: the RRT/ODT fit. */
const acesFit = (value: TSLNode): TSLNode =>
  value
    .mul(value.add(fr(0.0245786)))
    .sub(fr(0.000090537))
    .div(value.mul(fr(0.983729).mul(value).add(fr(0.432951))).add(fr(0.238081)));

/**
 * `display_transform`: into the ACES working space, the RRT/ODT fit, back out, then sRGB - the operator three's
 * `ACESFilmicToneMapping` (exposure 1) followed by the sRGB output transform applies.
 */
export const nrDisplayTransform = Fn(([scene]: [TSLNode]) => {
  const input = row3(
    [0.59719, 0.35458, 0.04823],
    [0.076, 0.90834, 0.01566],
    [0.0284, 0.13383, 0.83777],
    scene.div(fr(0.6)),
  ).toVar();
  const fitted = vec3(acesFit(input.x), acesFit(input.y), acesFit(input.z));
  const color = clamp(
    row3([1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602], fitted),
    vec3(f(0)),
    vec3(f(1)),
  ).toVar();
  return vec3(nrSrgbEncode(color.x), nrSrgbEncode(color.y), nrSrgbEncode(color.z));
}).setLayout(vec3Layout('nr_display_transform', ['scene']));

// ---------------------------------------------------------------------------------------------------------------
// The conditioning styles: exposure, contrast and saturation in HSL (frame.wgsl `nr_hsl` / `nr_hue` / `nr_rgb` /
// `nr_style`). The reference spells its fused multiply-adds as `fma`; so does this port (`nrFma`).

const ONE_SIXTH = 0x3e2aaaab;
const ONE_THIRD = 0x3eaaaaab;
const TWO_THIRDS = 0x3f2aaaab;

export const nrHsl = Fn(([rgb]: [TSLNode]) => {
  const high = max(max(rgb.x, rgb.y), rgb.z).toVar();
  const low = min(min(rgb.x, rgb.y), rgb.z).toVar();
  const sum = high.add(low).toVar();
  const light = sum.mul(f(0.5)).toVar();
  const hue = f(0).toVar();
  const saturation = f(0).toVar();
  If(high.greaterThan(low), () => {
    const delta = high.sub(low).toVar();
    const inverse = f(1).div(delta).toVar();
    saturation.assign(pick(light.greaterThan(f(0.5)), delta.div(f(2).sub(high).sub(low)), delta.div(sum)));
    If(high.equal(rgb.x), () => {
      hue.assign(nrFma(rgb.y.sub(rgb.z), inverse, pick(rgb.y.lessThan(rgb.z), f(6), f(0))).div(f(6)));
    })
      .ElseIf(high.equal(rgb.y), () => {
        hue.assign(nrFma(rgb.z.sub(rgb.x), inverse, f(2)).div(f(6)));
      })
      .Else(() => {
        hue.assign(nrFma(rgb.x.sub(rgb.y), inverse, f(4)).div(f(6)));
      });
  });
  return vec3(hue, saturation, light);
}).setLayout(vec3Layout('nr_hsl', ['rgb']));

export const nrHue = Fn(([p, q, hue]: [TSLNode, TSLNode, TSLNode]) => {
  const h = float(hue).toVar();
  If(h.lessThan(f(0)), () => {
    h.addAssign(f(1));
  });
  If(h.greaterThan(f(1)), () => {
    h.subAssign(f(1));
  });
  const result = float(p).toVar();
  If(h.lessThan(fBits(ONE_SIXTH)), () => {
    result.assign(nrFma(h, q.sub(p).mul(f(6)), p));
  })
    .ElseIf(h.lessThan(f(0.5)), () => {
      result.assign(q);
    })
    .ElseIf(h.lessThan(fBits(TWO_THIRDS)), () => {
      result.assign(nrFma(fBits(TWO_THIRDS).sub(h).mul(q.sub(p)), f(6), p));
    });
  return result;
}).setLayout({
  name: 'nr_hue',
  type: 'float',
  inputs: [
    { name: 'p', type: 'float' },
    { name: 'q', type: 'float' },
    { name: 'hue', type: 'float' },
  ],
});

export const nrRgb = Fn(([hsl]: [TSLNode]) => {
  const result = vec3(hsl.z).toVar();
  If(hsl.y.lessThanEqual(f(0)), () => {
    result.assign(vec3(hsl.z));
  }).Else(() => {
    const l = hsl.z;
    const s = hsl.y;
    const q = pick(l.lessThan(f(0.5)), l.mul(s.add(f(1))), nrFma(l.negate(), s, l.add(s)));
    const p = l.add(l).sub(q).toVar();
    result.assign(
      vec3(nrHue(p, q, hsl.x.add(fBits(ONE_THIRD))), nrHue(p, q, hsl.x), nrHue(p, q, hsl.x.sub(fBits(ONE_THIRD)))),
    );
  });
  return result;
}).setLayout(vec3Layout('nr_rgb', ['hsl']));

export const nrStyle = Fn(([neural, style, tone]: [TSLNode, TSLNode, TSLNode]) => {
  const styleTone = clamp(tone, f(0), f(1)).toVar();
  const cinematic = style.equal(f(1));
  const exposure = pick(cinematic, fr(-0.1).mul(styleTone), f(0));
  const contrast = pick(cinematic, fr(-0.25).mul(styleTone), f(0));
  const saturation = pick(cinematic, fr(-0.1).mul(styleTone), fr(-0.15).mul(styleTone));
  const scale = exp2(exposure).toVar();
  const channel = (component: TSLNode): TSLNode => {
    const value = clamp(component, f(0), f(1));
    const exposed = clamp(value.mul(scale), f(0), f(1)).toVar();
    const square = exposed.mul(exposed);
    const curveDelta = nrFma(square, f(3).sub(exposed.add(exposed)), exposed.negate());
    const curved = clamp(nrFma(contrast, curveDelta, exposed), f(0), f(1));
    return exp2(log2(max(f(0), exp2(log2(curved)))));
  };
  const color = vec3(channel(neural.x), channel(neural.y), channel(neural.z));
  const hsl = nrHsl(color).toVar();
  const adjusted = nrRgb(vec3(hsl.x, clamp(hsl.y.mul(saturation.add(f(1))), f(0), f(1)), hsl.z));
  const again = nrHsl(adjusted).toVar();
  return clamp(nrRgb(vec3(again.x, clamp(exp2(log2(again.y)), f(0), f(1)), again.z)), vec3(f(0)), vec3(f(1)));
}).setLayout({
  name: 'nr_style',
  type: 'vec3',
  inputs: [
    { name: 'neural', type: 'vec3' },
    { name: 'style', type: 'float' },
    { name: 'tone', type: 'float' },
  ],
});

/** Intensity dials the correction back towards the proxy the network started from, truncated to the half grid. */
const dial = (intensity: TSLNode, styled: TSLNode, code: TSLNode): TSLNode =>
  nrF16ToF32(nrTruncateHalf(clamp(nrFma(intensity, styled.sub(code), code), f(0), f(1))));

// ---------------------------------------------------------------------------------------------------------------

export interface ComposeSpec extends FrameGeometry {
  /** Pixels per row of `image` (frame.wgsl `image_pitch`); default `validWidth`. */
  imagePitch?: number;
  /** Dispatch label; default `'compose'`. */
  label?: string;
}

export interface ComposeBuffers {
  color: FrameColorSource;
  motion: FrameMotionSource;
  /** This frame's history (what input features read): rgba16float `validWidth * validHeight`. */
  history: BufferSource;
  /** Where the next history is written; never the same buffer as `history` (`NRHistory` ping-pongs them). */
  nextHistory: BufferSource;
  /** The network's head, f32 `[fullWidth * fullHeight][4]`. */
  head: NRTensor;
  params: NRFrameParams;
  /** Linear HDR output (a three.js `StorageTexture`, at least `validWidth x validHeight`). */
  outputTexture?: ThreeTexture;
  /** The reference's packed bgra8 canvas buffer (display transform applied), `imagePitch * validHeight` words. */
  image?: BufferSource;
}

/** `compose` (frame.wgsl): one invocation per valid pixel, workgroups of 8x8. */
export function createCompose(spec: ComposeSpec, buffers: ComposeBuffers): NRKernel {
  const { fullWidth, fullHeight, validWidth, validHeight } = spec;
  const label = spec.label ?? 'compose';
  const pitch = spec.imagePitch ?? validWidth;
  const { color, motion, history, nextHistory, head, params, outputTexture, image } = buffers;
  if (head.format !== 'f32' || head.channels !== 4 || head.allocRows < fullWidth * fullHeight) {
    throw new Error(`${label}: head must be f32 [${fullWidth * fullHeight}][4]`);
  }
  if (validWidth > fullWidth || validHeight > fullHeight) throw new Error(`${label}: valid exceeds the field`);
  for (const [name, buffer] of [
    ['history', history],
    ['nextHistory', nextHistory],
  ] as const) {
    if (buffer.attribute.count < validWidth * validHeight * 2) throw new Error(`${label}: ${name} is too small`);
  }
  if (image && image.attribute.count < pitch * validHeight) throw new Error(`${label}: image is too small`);
  if (pitch < validWidth) throw new Error(`${label}: imagePitch ${pitch} is below the width`);
  const readers = frameReaders(spec, color, motion);
  const outputs: Record<string, BufferSource> = image ? { nextHistory, image } : { nextHistory };
  return kernel({
    label,
    kind: 'compose',
    workgroupSize: [8, 8],
    dispatch: [Math.ceil(validWidth / 8), Math.ceil(validHeight / 8)],
    inputs: { ...readers.inputs, history, head },
    outputs,
    body: (views) => {
      const v = views as Record<string, TSLNode>;
      const sampler = historySampler(v.history, validWidth, validHeight);
      const x = workgroupId.x.mul(u(8)).add(localId.x).toVar();
      const y = workgroupId.y.mul(u(8)).add(localId.y).toVar();

      const storeHistory = (pixel: TSLNode, rgb: TSLNode) => {
        const word = pixel.mul(u(2)).toVar();
        v.nextHistory.element(word).assign(nrTruncateHalf(rgb.x).bitOr(nrTruncateHalf(rgb.y).shiftLeft(u(16))));
        v.nextHistory.element(word.add(u(1))).assign(nrTruncateHalf(rgb.z).bitOr(u(0x3c00 << 16)));
      };
      const present = (rgb: TSLNode, linear: TSLNode) => {
        if (outputTexture) textureStore(outputTexture, uvec2(x, y), vec4(linear, f(1))).toStack();
        if (!image) return;
        const q = uvec3(clamp(round(nrDisplayTransform(rgb).mul(f(255))), vec3(f(0)), vec3(f(255)))).toVar();
        v.image.element(y.mul(u(pitch)).add(x)).assign(
          q.z
            .bitOr(q.y.shiftLeft(u(8)))
            .bitOr(q.x.shiftLeft(u(16)))
            .bitOr(u(0xff000000)),
        );
      };

      If(x.lessThan(u(validWidth)).and(y.lessThan(u(validHeight))), () => {
        const pixel = y.mul(u(validWidth)).add(x).toVar();
        const field = y.mul(u(fullWidth)).add(x).mul(u(4)).toVar();
        const scene = readers.sceneAt(v, x, y).toVar();
        const paper = max(params.paperWhite, fr(0.05)).toVar();
        const original = scene.div(paper).toVar();
        const proxy = (c: TSLNode) => nrProxyComponent(c, params.paperWhite);
        const code = vec3(proxy(scene.x), proxy(scene.y), proxy(scene.z)).toVar();

        If(params.enabled.equal(u(0)), () => {
          // NR off: the rendered frame through the same output path, and the history kept primed with the proxy.
          storeHistory(pixel, code);
          present(scene, scene);
        }).Else(() => {
          const residual = vec3(
            loadF32(v.head, field),
            loadF32(v.head, field.add(u(1))),
            loadF32(v.head, field.add(u(2))),
          );
          const neural = clamp(code.add(residual.mul(f(0.25))), vec3(f(0)), vec3(f(1))).toVar();
          const motionUv = readers.motionAt(v, x, y).toVar();
          let blend = params.historyValid.notEqual(u(0));
          if (spec.rejectOffscreenHistory) blend = blend.and(historyOnScreen(x, y, motionUv, validWidth, validHeight));
          If(blend, () => {
            const logit = loadF32(v.head, field.add(u(3)));
            const sigmoid = f(1).div(f(1).add(exp2(logit.mul(fr(-Math.LOG2E)))));
            const weight = clamp(sigmoid.mul(params.blendScale), f(0), f(1));
            const previous = sampler.reproject(x, y, motionUv);
            neural.assign(neural.add(previous.sub(neural).mul(weight)));
          });
          // The history is the blended code value and nothing downstream of it.
          const stored = [neural.x, neural.y, neural.z].map((c) => nrTruncateHalf(c).toVar());
          const word = pixel.mul(u(2)).toVar();
          v.nextHistory.element(word).assign(stored[0].bitOr(stored[1].shiftLeft(u(16))));
          v.nextHistory.element(word.add(u(1))).assign(stored[2].bitOr(u(0x3c00 << 16)));
          const truncated = vec3(nrF16ToF32(stored[0]), nrF16ToF32(stored[1]), nrF16ToF32(stored[2])).toVar();

          const styled = vec3(neural).toVar();
          If(params.style.notEqual(f(0)), () => {
            styled.assign(nrStyle(truncated, params.style, params.localTone));
          });
          const published = vec3(truncated).toVar();
          If(params.style.notEqual(f(0)).or(params.intensity.notEqual(f(1))), () => {
            const k = params.intensity;
            published.assign(vec3(dial(k, styled.x, code.x), dial(k, styled.y, code.y), dial(k, styled.z, code.z)));
          });

          // Back out of code space onto the scene the frame arrived in.
          const proxyLinear = vec3(nrSrgbDecode(code.x), nrSrgbDecode(code.y), nrSrgbDecode(code.z));
          const neuralLinear = vec3(nrSrgbDecode(published.x), nrSrgbDecode(published.y), nrSrgbDecode(published.z));
          const upgraded = nrUpgradeToneMap(original, proxyLinear, neuralLinear).toVar();
          const originalY = nrLuminance(original).toVar();
          const ratio = pick(originalY.equal(f(0)), f(1), clamp(nrLuminance(upgraded).div(originalY), f(0), f(4)));
          const luminanceOnly = original.mul(ratio).toVar();
          const result = luminanceOnly.add(upgraded.sub(luminanceOnly).mul(params.colorStrength)).mul(paper).toVar();
          present(result, result);
        });
      });
    },
  });
}
