// The temporal loop: two history buffers alternating by frame parity, and the frame kernels built once per parity.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The reference WebGPU demo records its graph once,
// so it reads one history buffer, writes the other and copies them back (production-pipeline.js, 'history swap').
// Here the kernels are prebuilt for both parities instead (design 4.4), which is what the native pipeline does
// (docs/frame.md, "Frames in flight"): frame N runs `inputFeatures[N & 1]`, the network, then `compose[N & 1]`.
//
// History format: rgba16float, two u32 words per valid pixel, rgb = the composed code value truncated to the half
// grid, a = 1.0.

import { wordAttribute } from '../tensors.js';
import type { BufferSource } from '../tsl/KernelBuilder.js';
import type { NRKernel, NRTensor } from '../types.js';
import { createCompose, type ComposeSpec } from './compose.js';
import type { FrameColorSource, FrameMotionSource, NRFrameParams, ThreeTexture } from './frameInputs.js';
import { createInputFeatures } from './inputFeatures.js';

/** Frame parity: which history buffer a frame reads. */
export type Parity = 0 | 1;

/** Two rgba16float history buffers of `width x height` pixels (zero-filled; the first frame has no history). */
export class NRHistory {
  readonly buffers: readonly [BufferSource, BufferSource];

  constructor(
    readonly width: number,
    readonly height: number,
  ) {
    this.buffers = [{ attribute: wordAttribute(width * height * 8) }, { attribute: wordAttribute(width * height * 8) }];
  }

  /** The history frame `parity` reads (input features, compose's blend). */
  read(parity: Parity): BufferSource {
    return this.buffers[parity];
  }

  /** The history frame `parity` writes (compose), which frame `parity ^ 1` reads. */
  write(parity: Parity): BufferSource {
    return this.buffers[parity ^ 1];
  }
}

export interface FrameKernelsBuffers {
  color: FrameColorSource;
  motion: FrameMotionSource;
  /** The network's input features, f32 `[fullRows][16]`. */
  features: NRTensor;
  /** The network's head, f32 `[fullRows][4]`. */
  head: NRTensor;
  params: NRFrameParams;
  history: NRHistory;
  /** Linear HDR output texture (see `createCompose`). */
  outputTexture?: ThreeTexture;
  /** Packed bgra8 image buffer (see `createCompose`). */
  image?: BufferSource;
}

/** The frame kernels for both parities. */
export interface FrameKernels {
  /** Before the network: `inputFeatures[parity]`. */
  inputFeatures: readonly [NRKernel, NRKernel];
  /** After the network: `compose[parity]`. */
  compose: readonly [NRKernel, NRKernel];
}

/**
 * Build `input_features` and `compose` for both history parities. A frame is
 * `renderer.compute([inputFeatures[p].node, ...network nodes, compose[p].node])` with `p = frameIndex & 1`; set
 * `params` (seed, historyValid, ...) before it.
 */
export function createFrameKernels(spec: ComposeSpec, buffers: FrameKernelsBuffers): FrameKernels {
  const { history, ...rest } = buffers;
  if (history.width !== spec.validWidth || history.height !== spec.validHeight) {
    throw new Error(`history is ${history.width}x${history.height}, the frame ${spec.validWidth}x${spec.validHeight}`);
  }
  const parities: Parity[] = [0, 1];
  const inputFeatures = parities.map((parity) =>
    createInputFeatures({ ...spec, label: undefined }, { ...rest, history: history.read(parity) }),
  ) as [NRKernel, NRKernel];
  const compose = parities.map((parity) =>
    createCompose(
      { ...spec, label: undefined },
      { ...rest, history: history.read(parity), nextHistory: history.write(parity) },
    ),
  ) as [NRKernel, NRKernel];
  return { inputFeatures, compose };
}
