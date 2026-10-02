// Port of OpenDLSS-NR ports/browser-webgpu/src/geometry.js (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/src/geometry.js).
//
// The shapes the graph is built on: the padded field, the six pooling levels, the window phase cycle, and the byte
// layout of a block's weight tensor. None of this is a free choice - the field size decides which tokens exist, and
// therefore the result inside the valid rectangle as well (see the reference's docs/network.md).

export const alignUp = (value: number, alignment: number): number => Math.ceil(value / alignment) * alignment;

/** One pooled resolution level. */
export interface NRLevel {
  width: number;
  height: number;
  rows: number;
}

/** The padded field and pooled levels of one valid image size (`geometryFromValid`). */
export interface NRGeometry {
  validWidth: number;
  validHeight: number;
  fullWidth: number;
  fullHeight: number;
  /** The six pooled levels, level 0 (half the field) first. */
  levels: NRLevel[];
  fullRows: number;
  /** Tokens of the ViT (`levels[5].rows`). */
  vitTokens: number;
  /** `vitTokens` rounded up to 64. */
  paddedVitTokens: number;
}

/**
 * Every level halves its input and rounds up to 4, and the decoder doubles the chain back up, so a dimension needs
 * enough headroom for all of those halvings to be exact: the valid size aligned to two to the power of the number of
 * size reductions the graph makes on that axis. Six are the halvings; level 0 adds a seventh when it is not a whole
 * number of 8-pixel windows.
 */
function fieldAlignment(valid: number): number {
  let reductions = 0;
  let size = valid;
  for (let level = 0; level < 6; ++level) {
    const half = alignUp(Math.floor((size + 1) / 2), 4);
    if (half < size) reductions += 1;
    if (level === 0 && half % 8 !== 0) reductions += 1;
    size = half;
  }
  return 1 << reductions;
}

export function geometryFromValid(validWidth: number, validHeight: number): NRGeometry {
  const alignWidth = fieldAlignment(validWidth);
  const alignHeight = fieldAlignment(validHeight);
  let fullWidth = Math.max(320, alignUp(validWidth, alignWidth));
  const fullHeight = Math.max(320, alignUp(validHeight, alignHeight));
  // One more alignment step on the width when both axes are a multiple of four alignments. The rule has no stated
  // reason; it has to be reproduced because it moves the window grid.
  if (fullWidth % (4 * alignWidth) === 0 && fullHeight % (4 * alignHeight) === 0) fullWidth += alignWidth;

  const levels: NRLevel[] = [];
  let width = fullWidth;
  let height = fullHeight;
  for (let level = 0; level < 6; ++level) {
    width = alignUp(Math.floor((width + 1) / 2), 4);
    height = alignUp(Math.floor((height + 1) / 2), 4);
    levels.push({ width, height, rows: width * height });
  }
  if (levels[0].width % 8 || levels[0].height % 8) {
    throw new Error(
      `unsupported size ${validWidth}x${validHeight}: level 0 ` +
        `(${levels[0].width}x${levels[0].height}) is not a whole number of 8-pixel windows; ` +
        'use at least 33 pixels on each axis',
    );
  }
  const vitTokens = levels[5].rows;
  return {
    validWidth,
    validHeight,
    fullWidth,
    fullHeight,
    levels,
    fullRows: fullWidth * fullHeight,
    vitTokens,
    paddedVitTokens: (vitTokens + 63) & ~63,
  };
}

/** The four window views, as origin offsets `(-shiftX, -shiftY)`. */
const PHASES: readonly (readonly [number, number])[] = [
  [0, 0],
  [4, 4],
  [4, 0],
  [0, 4],
];

/** `[shiftX, shiftY]` of a window phase index (the index cycles mod 4). */
export const windowPhase = (index: number): readonly [number, number] => PHASES[index & 3];

/**
 * Per-level window phase counters. Level 6 is the un-pooled field (blocks 0 and 70); 0..5 are the pooled levels.
 * Every level runs its own cycle, one step per block at that level in visit order, and a decoder stage continues
 * the count its encoder stage left.
 */
export class WindowPhases {
  readonly counters = new Int32Array(7);
  take(level: number): number {
    return this.counters[level]++;
  }
  reset(): void {
    this.counters.fill(0);
  }
}

const standardHidden = (channels: number): number => {
  if (channels === 32 || channels === 64 || channels === 128 || channels === 256) return 128;
  throw new Error(`no fused layout for ${channels} channels`);
};

/** Byte offsets inside one block's weight tensor (FFN -> QKV -> window attention -> projection). */
export interface FusedLayout {
  hidden: number;
  heads: number;
  expertFfn: boolean;
  expertCount: number;
  expand: number;
  contractWeights: number;
  ffnCosSkip: number;
  qkv: number;
  relative: number;
  scale: number;
  projection: number;
  attnCosSkip: number;
  endWithoutPadding: number;
  /** Block 0 only: the 16 -> 32 f16 input adapter. */
  inputAdapter?: number;
  /** First decoder block only: the 2x upsample weight. */
  upsampleWeight?: number;
  /** First decoder block only: the transition skip scale. */
  transitionScale?: number;
  /** Block 70 only: the post blend's input scale. */
  inputScale?: number;
  /** Block 70 only: the post blend's adapter scale. */
  adapterScale?: number;
  /** Block 70 only: the 32 -> 4 f16 head. */
  postWeights?: number;
}

/**
 * Byte offsets inside one block's weight tensor. A block is FFN -> QKV -> window attention -> projection, and the
 * tensor holds the matrices in that order with two 16-byte pads whose purpose is not visible.
 */
export function fusedLayout(channels: number, base = 0): FusedLayout {
  const hidden = standardHidden(channels);
  const heads = channels / 32;
  const expertFfn = channels >= 64;
  const expertCount = expertFfn ? channels / 32 : 0;
  const expandBytes = expertFfn ? expertCount * channels * 128 : channels * hidden;
  const ffnWeightBytes = expertFfn
    ? expandBytes + expertCount * 128 * 32 + expertCount * 32 * channels
    : expandBytes + hidden * channels;
  const ffnCosSkip = base + ffnWeightBytes + 16;
  const qkv = ffnCosSkip + channels * 2 + 16;
  const relative = qkv + channels * channels * 3;
  const scale = relative + heads * 8192;
  const projection = scale + alignUp(heads * 4, 16);
  const attnCosSkip = projection + channels * channels;
  return {
    hidden,
    heads,
    expertFfn,
    expertCount,
    expand: base,
    contractWeights: base + expandBytes,
    ffnCosSkip,
    qkv,
    relative,
    scale,
    projection,
    attnCosSkip,
    endWithoutPadding: attnCosSkip + channels * 2,
  };
}

/** Block 0: the same block with a 16 -> 32 f16 input adapter in front of it. */
export const preFusedLayout = (): FusedLayout => ({
  hidden: 128,
  heads: 1,
  expertFfn: false,
  expertCount: 0,
  expand: 0,
  contractWeights: 4096,
  inputAdapter: 8208,
  ffnCosSkip: 9232,
  qkv: 9312,
  relative: 12384,
  scale: 20576,
  projection: 20592,
  attnCosSkip: 21616,
  endWithoutPadding: 21680,
});

/** The first block of a decoder stage: the 2x upsample weight and the skip scale sit before the QKV. */
export function upsampleFusedLayout(inputChannels: number, channels: number): FusedLayout {
  if (inputChannels !== channels * 2) throw new Error('upsample layout expects 2x input channels');
  const hidden = standardHidden(channels);
  const heads = channels / 32;
  const narrowPadding = channels === 32 ? 16 : 0;
  const expertFfn = channels >= 64;
  const expertCount = expertFfn ? channels / 32 : 0;
  const expandBytes = expertFfn ? expertCount * channels * 128 : channels * hidden;
  const ffnWeightBytes = expertFfn
    ? expandBytes + expertCount * 128 * 32 + expertCount * 32 * channels
    : expandBytes + hidden * channels;
  const upsampleWeight = ffnWeightBytes;
  const ffnCosSkip = upsampleWeight + inputChannels * channels + narrowPadding;
  const transitionScale = ffnCosSkip + channels * 2 + narrowPadding;
  const qkv = transitionScale + channels * 2;
  const relative = qkv + channels * channels * 3;
  const scale = relative + heads * 8192;
  const projection = scale + alignUp(heads * 4, 16);
  const attnCosSkip = projection + channels * channels;
  return {
    hidden,
    heads,
    expertFfn,
    expertCount,
    expand: 0,
    contractWeights: expandBytes,
    upsampleWeight,
    ffnCosSkip,
    transitionScale,
    qkv,
    relative,
    scale,
    projection,
    attnCosSkip,
    endWithoutPadding: attnCosSkip + channels * 2,
  };
}

/** Block 70: the post blend's two scales in front, the 32 -> 4 f16 head at the end. */
export const postFusedLayout = (): FusedLayout => ({
  hidden: 128,
  heads: 1,
  expertFfn: false,
  expertCount: 0,
  expand: 0,
  contractWeights: 4096,
  ffnCosSkip: 8208,
  inputScale: 8272,
  adapterScale: 8336,
  qkv: 8400,
  relative: 11472,
  scale: 19664,
  projection: 19680,
  attnCosSkip: 20704,
  postWeights: 20784,
  endWithoutPadding: 21808,
});

const MAX_GROUPS = 65535;

/**
 * A 1D dispatch of `count` invocations at 64 per workgroup, folded into two dimensions past the 65535-group limit
 * (`graph.js` `grid1d`). Returns `[x, y, 1]`.
 */
export function grid1d(count: number): [number, number, number] {
  const groups = Math.ceil(count / 64);
  if (groups <= MAX_GROUPS) return [groups, 1, 1];
  return [MAX_GROUPS, Math.ceil(groups / MAX_GROUPS), 1];
}
