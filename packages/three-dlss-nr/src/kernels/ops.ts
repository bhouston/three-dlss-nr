// The elementwise steps between the matrix multiplies, as TSL compute kernels.
//
// Port of OpenDLSS-NR ports/browser-webgpu/shaders/ops.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/ops.wgsl) to three.js TSL, plus
// a word copy that replaces the reference's pass-splitting `copyBufferToBuffer` captures (design 4.2).
//
// Same arithmetic, same rounding schedule, same thread mapping as the reference: one invocation per four consecutive
// output values (one E4 word, two half words), folded past 65535 workgroups exactly as `linear_index` folds them.
// What differs is the plumbing: the reference's `Params` uniform becomes JS constants baked into each node graph, the
// dual flag becomes the presence of `outputF16`, and the aux buffer is a `HalfVector` whose offsets (`aux_a`, `aux_b`)
// are spec fields.
//
// Rounding (Appendix A.4 of the design; every value published with `encode_e4m3(f16_bits(v))`, which keeps -0):
//   downsample         roundF16(roundF16(roundF16(a + b) + roundF16(c + d)) * 0.25)
//   upsample_residual  roundF16(up + skip * scale)                 (the product is exact; one rounding)
//   post_blend         roundF16(roundF16(in * sA) + skip * sB)

import { If, localId, workgroupId } from 'three/tsl';

import { grid1d } from '../geometry.js';
import { kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { nrDecodeE4m3, nrEncodeE4m3, nrF16Bits, nrF16ToF32, nrRoundF16 } from '../tsl/numerics.js';
import { f, loadE4, loadF32, loadHalfBits, packHalfPair, packWord4, u, type TSLNode } from '../tsl/packed.js';
import type { HalfVector, NRKernel, NRTensor, TensorFormat } from '../types.js';

/** `convert_f32_to_f16`: the f32 input features published to the half grid (RNE). */
export interface ConvertF32ToF16Spec {
  /** Values to convert (a multiple of 4); the reference passes `fullRows * 16`. */
  count: number;
  label: string;
}

export interface ConvertF32ToF16Buffers {
  /** f32 values. */
  input: NRTensor;
  /** Half values, same index. */
  output: NRTensor;
}

/** A 2x2 pool / upsample between two levels of `[height][width][channels]` row-major pixels. */
export interface LevelPairSpec {
  /** Channels per pixel (a multiple of 4). */
  channels: number;
  inWidth: number;
  inHeight: number;
  outWidth: number;
  outHeight: number;
  /** Output values to write; default `outWidth * outHeight * channels` (what the graph always passes). */
  count?: number;
  label: string;
}

/** `downsample`: 2x2 box pool of a half tensor into an E4 tensor. */
export type DownsampleSpec = LevelPairSpec;

export interface DownsampleBuffers {
  /** Half `[inHeight][inWidth][channels]`. */
  input: NRTensor;
  /** E4 `[outHeight][outWidth][channels]`. */
  output: NRTensor;
}

/** `upsample_residual`: nearest 2x upsample of a half tensor plus an E4 skip times a per-channel half scale. */
export interface UpsampleResidualSpec extends LevelPairSpec {
  /** Half index of the first scale in `scale` (the reference's `aux_a`); default 0. */
  scaleOffset?: number;
}

export interface UpsampleResidualBuffers {
  /** Half `[inHeight][inWidth][channels]` (the low level). */
  input: NRTensor;
  /** E4 skip `[outHeight][outWidth][channels]`. */
  skip: NRTensor;
  /** Per-channel half scales (`model.auxVector`). */
  scale: HalfVector;
  /** E4 output, same layout as `skip`. */
  output: NRTensor;
  /** Optional raw half output at the same index (the reference's dual flag). */
  outputF16?: NRTensor;
}

/** `post_blend`: the level-0 decoder output upsampled and scaled, plus block 0's output scaled. */
export interface PostBlendSpec extends LevelPairSpec {
  /** Half index of the input scales in `scales` (the reference's `aux_a`); default 0. */
  inputScaleOffset?: number;
  /** Half index of the skip scales in `scales` (the reference's `aux_b`); default `channels`. */
  skipScaleOffset?: number;
}

export interface PostBlendBuffers {
  /** E4 `[inHeight][inWidth][channels]` (the low level). */
  input: NRTensor;
  /** E4 skip `[outHeight][outWidth][channels]`. */
  skip: NRTensor;
  /** Input scales then skip scales (`model.auxPair`). */
  scales: HalfVector;
  output: NRTensor;
  outputF16?: NRTensor;
}

/** A word copy (boundary capture). */
export interface CopyWordsSpec {
  /** Words to copy; default: every word of `source`. */
  words?: number;
  label: string;
}

export interface CopyWordsBuffers {
  source: BufferSource;
  target: BufferSource;
}

// ---------------------------------------------------------------------------------------------------------------

const check = (condition: boolean, label: string, message: string): void => {
  if (!condition) throw new Error(`${label}: ${message}`);
};

const checkFormat = (tensor: NRTensor, format: TensorFormat, what: string, label: string): void =>
  check(tensor.format === format, label, `${what} "${tensor.label}" is ${tensor.format}, expected ${format}`);

/** Values a tensor's allocation holds (padding rows included). */
const capacity = (tensor: NRTensor): number => tensor.allocRows * tensor.channels;

/** The first of the four values this invocation owns (`linear_index(id) * 4`). */
const quadBase = (): TSLNode =>
  workgroupId.x
    .add(workgroupId.y.mul(u(65535)))
    .mul(u(64))
    .add(localId.x)
    .mul(u(4));

/** Half `index` of a packed half buffer as an f32, on the bit pattern (`half_at` / `aux_half`). */
const halfAt = (buffer: TSLNode, index: TSLNode): TSLNode => nrF16ToF32(loadHalfBits(buffer, index));

/** E4 byte `index` of a packed byte buffer, decoded (`e4_at`). */
const e4At = (buffer: TSLNode, index: TSLNode): TSLNode => nrDecodeE4m3(loadE4(buffer, index));

/** ops.wgsl `publish`: one E4 word, and two half words when there is a raw half output. */
function publish(base: TSLNode, values: readonly TSLNode[], outE4: TSLNode, outF16: TSLNode | undefined): void {
  const halves = values.map((value) => nrF16Bits(value).toVar());
  outE4
    .element(base.shiftRight(u(2)))
    .assign(packWord4(...(halves.map((h) => nrEncodeE4m3(h)) as [TSLNode, TSLNode, TSLNode, TSLNode])));
  if (outF16) {
    const word = base.shiftRight(u(1));
    outF16.element(word).assign(packHalfPair(halves[0], halves[1]));
    outF16.element(word.add(u(1))).assign(packHalfPair(halves[2], halves[3]));
  }
}

function levelCount(spec: LevelPairSpec): number {
  const count = spec.count ?? spec.outWidth * spec.outHeight * spec.channels;
  check(spec.channels % 4 === 0 && spec.channels > 0, spec.label, `channels ${spec.channels} is not a multiple of 4`);
  check(count % 4 === 0 && count > 0, spec.label, `count ${count} is not a positive multiple of 4`);
  check(count <= 0xffffffff - 64 * 4, spec.label, `count ${count} overflows u32 indexing`);
  return count;
}

/** The source index of the low-level pixel under an output pixel (`upsample_residual` / `post_blend`). */
function upsampleSource(pixel: TSLNode, spec: LevelPairSpec): TSLNode {
  const sy = pixel.div(u(spec.outWidth)).shiftRight(u(1));
  const sx = pixel.mod(u(spec.outWidth)).shiftRight(u(1));
  return sy.mul(u(spec.inWidth)).add(sx).mul(u(spec.channels));
}

// ---------------------------------------------------------------------------------------------------------------

/** `convert_f32_to_f16` (ops.wgsl): the f32 input features published to the half grid for the adapter. */
export function createConvertF32ToF16(spec: ConvertF32ToF16Spec, buffers: ConvertF32ToF16Buffers): NRKernel {
  const { count, label } = spec;
  const { input, output } = buffers;
  checkFormat(input, 'f32', 'input', label);
  checkFormat(output, 'f16', 'output', label);
  check(count % 4 === 0 && count > 0, label, `count ${count} is not a positive multiple of 4`);
  check(count <= capacity(input) && count <= capacity(output), label, `count ${count} exceeds a tensor`);
  return kernel({
    label,
    kind: 'convert_f32_to_f16',
    workgroupSize: [64],
    dispatch: grid1d(count / 4),
    inputs: { inF32: input },
    outputs: { outF16: output },
    body: ({ inF32, outF16 }) => {
      const base = quadBase().toVar();
      If(base.lessThan(u(count)), () => {
        // f16_bits of the stored f32, read as bits (no f32 load that a backend could canonicalize).
        const half = (k: number) => nrF16Bits(loadF32(inF32, base.add(u(k))));
        const word = base.shiftRight(u(1));
        outF16.element(word).assign(packHalfPair(half(0), half(1)));
        outF16.element(word.add(u(1))).assign(packHalfPair(half(2), half(3)));
      });
    },
  });
}

/** `downsample` (ops.wgsl): the 2x2 box pool that takes a stage down a level, `((a+b)+(c+d)) * 0.25` in halves. */
export function createDownsample(spec: DownsampleSpec, buffers: DownsampleBuffers): NRKernel {
  const { label, channels, inWidth, inHeight, outWidth } = spec;
  const { input, output } = buffers;
  checkFormat(input, 'f16', 'input', label);
  checkFormat(output, 'e4', 'output', label);
  const count = levelCount(spec);
  check(count <= capacity(output), label, `count ${count} exceeds the output`);
  check(inWidth * inHeight * channels <= capacity(input), label, 'the input level exceeds the input tensor');
  return kernel({
    label,
    kind: 'downsample',
    workgroupSize: [64],
    dispatch: grid1d(count / 4),
    inputs: { inF16: input },
    outputs: { outE4: output },
    body: ({ inF16, outE4 }) => {
      const base = quadBase().toVar();
      If(base.lessThan(u(count)), () => {
        const c = base.mod(u(channels));
        const pixel = base.div(u(channels)).toVar();
        const sx = pixel.mod(u(outWidth)).mul(u(2)).toVar();
        const sy = pixel.div(u(outWidth)).mul(u(2)).toVar();
        const values = [0, 1, 2, 3].map(() => f(0).toVar());
        If(
          sx
            .add(u(1))
            .lessThan(u(inWidth))
            .and(sy.add(u(1)).lessThan(u(inHeight))),
          () => {
            const i00 = sy.mul(u(inWidth)).add(sx).mul(u(channels)).add(c).toVar();
            const i10 = i00.add(u(channels)).toVar();
            const i01 = sy.add(u(1)).mul(u(inWidth)).add(sx).mul(u(channels)).add(c).toVar();
            const i11 = i01.add(u(channels)).toVar();
            values.forEach((value, k) => {
              const top = nrRoundF16(halfAt(inF16, i00.add(u(k))).add(halfAt(inF16, i10.add(u(k)))));
              const bottom = nrRoundF16(halfAt(inF16, i01.add(u(k))).add(halfAt(inF16, i11.add(u(k)))));
              value.assign(nrRoundF16(nrRoundF16(top.add(bottom)).mul(f(0.25))));
            });
          },
        );
        publish(base, values, outE4, undefined);
      });
    },
  });
}

/**
 * `upsample_residual` (ops.wgsl): a decoder stage's entry - the level below doubled, plus the encoder skip times a
 * per-channel scale, rounded once. With `outputF16` it also publishes the raw halves (the reference's dual flag).
 */
export function createUpsampleResidual(spec: UpsampleResidualSpec, buffers: UpsampleResidualBuffers): NRKernel {
  const { label, channels } = spec;
  const { input, skip, scale, output, outputF16 } = buffers;
  const scaleOffset = spec.scaleOffset ?? 0;
  checkFormat(input, 'f16', 'input', label);
  checkFormat(skip, 'e4', 'skip', label);
  checkFormat(output, 'e4', 'output', label);
  if (outputF16) checkFormat(outputF16, 'f16', 'outputF16', label);
  const count = levelCount(spec);
  check(count <= capacity(output) && count <= capacity(skip), label, `count ${count} exceeds the output or skip`);
  if (outputF16) check(count <= capacity(outputF16), label, `count ${count} exceeds outputF16`);
  check(scaleOffset + channels <= scale.count, label, `scales [${scaleOffset}, +${channels}) exceed ${scale.count}`);
  return kernel({
    label,
    kind: 'upsample_residual',
    workgroupSize: [64],
    dispatch: grid1d(count / 4),
    inputs: { inF16: input, skipE4: skip, aux: scale },
    outputs: outputF16 ? { outE4: output, outF16: outputF16 } : { outE4: output },
    body: (views) => {
      const { inF16, skipE4, aux, outE4 } = views;
      const base = quadBase().toVar();
      If(base.lessThan(u(count)), () => {
        const c = base.mod(u(channels)).toVar();
        const source = upsampleSource(base.div(u(channels)), spec)
          .add(c)
          .toVar();
        const values = [0, 1, 2, 3].map((k) =>
          nrRoundF16(
            halfAt(inF16, source.add(u(k))).add(
              e4At(skipE4, base.add(u(k))).mul(halfAt(aux, c.add(u(scaleOffset + k)))),
            ),
          ),
        );
        publish(base, values, outE4, outputF16 ? (views as { outF16: TSLNode }).outF16 : undefined);
      });
    },
  });
}

/**
 * `post_blend` (ops.wgsl): the last merge before block 70 - the level-0 decoder output doubled and scaled (published
 * to half), plus block 0's output scaled (folded into the final rounding).
 */
export function createPostBlend(spec: PostBlendSpec, buffers: PostBlendBuffers): NRKernel {
  const { label, channels } = spec;
  const { input, skip, scales, output, outputF16 } = buffers;
  const auxA = spec.inputScaleOffset ?? 0;
  const auxB = spec.skipScaleOffset ?? channels;
  checkFormat(input, 'e4', 'input', label);
  checkFormat(skip, 'e4', 'skip', label);
  checkFormat(output, 'e4', 'output', label);
  if (outputF16) checkFormat(outputF16, 'f16', 'outputF16', label);
  const count = levelCount(spec);
  check(count <= capacity(output) && count <= capacity(skip), label, `count ${count} exceeds the output or skip`);
  if (outputF16) check(count <= capacity(outputF16), label, `count ${count} exceeds outputF16`);
  check(Math.max(auxA, auxB) + channels <= scales.count, label, `scales exceed ${scales.count} halves`);
  return kernel({
    label,
    kind: 'post_blend',
    workgroupSize: [64],
    dispatch: grid1d(count / 4),
    inputs: { inE4: input, skipE4: skip, aux: scales },
    outputs: outputF16 ? { outE4: output, outF16: outputF16 } : { outE4: output },
    body: (views) => {
      const { inE4, skipE4, aux, outE4 } = views;
      const base = quadBase().toVar();
      If(base.lessThan(u(count)), () => {
        const c = base.mod(u(channels)).toVar();
        const source = upsampleSource(base.div(u(channels)), spec)
          .add(c)
          .toVar();
        const values = [0, 1, 2, 3].map((k) => {
          const upsampled = nrRoundF16(e4At(inE4, source.add(u(k))).mul(halfAt(aux, c.add(u(auxA + k)))));
          return nrRoundF16(upsampled.add(e4At(skipE4, base.add(u(k))).mul(halfAt(aux, c.add(u(auxB + k))))));
        });
        publish(base, values, outE4, outputF16 ? (views as { outF16: TSLNode }).outF16 : undefined);
      });
    },
  });
}

/**
 * Copy words of one buffer into another inside the compute pass (design 4.2: boundary captures without splitting
 * the pass, unlike the reference's `copyBufferToBuffer`). One invocation per word.
 */
export function createCopyWords(spec: CopyWordsSpec, buffers: CopyWordsBuffers): NRKernel {
  const { label } = spec;
  const { source, target } = buffers;
  const words = spec.words ?? source.attribute.count;
  check(Number.isInteger(words) && words > 0, label, `words ${words} is not a positive integer`);
  check(words <= source.attribute.count && words <= target.attribute.count, label, `${words} words exceed a buffer`);
  return kernel({
    label,
    kind: 'copy_words',
    workgroupSize: [64],
    dispatch: grid1d(words),
    inputs: { source },
    outputs: { target },
    body: ({ source: from, target: to }) => {
      const index = workgroupId.x
        .add(workgroupId.y.mul(u(65535)))
        .mul(u(64))
        .add(localId.x)
        .toVar();
      If(index.lessThan(u(words)), () => {
        to.element(index).assign(from.element(index));
      });
    },
  });
}
