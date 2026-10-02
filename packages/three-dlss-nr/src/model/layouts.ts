// The 153 weight records of the 71-block network and what lies inside each one.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The record list and every offset below are derived
// from what the reference WebGPU port's graph reads (ports/browser-webgpu/src/graph.js `block`, `splitBlock`, `vit`,
// `record`) through the ported layout functions of geometry.js, and from the model directory layout in the
// reference's docs/weights.md. Nothing here is a free choice except the gains of the synthetic weights
// (`synthetic/gains.ts`): the regions say which bytes a kernel reads and as what.

import {
  alignUp,
  fusedLayout,
  postFusedLayout,
  preFusedLayout,
  upsampleFusedLayout,
  type FusedLayout,
} from '../geometry.js';

/** Blocks of the network (`manifest.totals.blockCount`). */
export const NR_BLOCK_COUNT = 71;

/** What an FP8 matrix is for; selects its synthetic gain. */
export type Fp8Role = 'expand' | 'contract' | 'merge' | 'qkv' | 'projection' | 'transition';

/** One typed region of a record. Offsets and lengths are in bytes from the start of the record. */
export type NRRegion =
  /** An E4M3 matrix in MMA fragment order: `k / batchK` matrices of `batchK x n`, `k * n` bytes. */
  | { kind: 'fp8'; role: Fp8Role; offset: number; k: number; n: number; batchK: number }
  /** An f16 matrix in `m16n8k16` fragment order (`packedF16WeightIndex`): the input adapter or the head. */
  | { kind: 'f16-matrix'; role: 'adapter' | 'head'; offset: number; k: number; n: number }
  /** A per-channel f16 scale vector (FFN / attention / transition / post-blend skip scales). */
  | { kind: 'skip-scale'; offset: number; count: number }
  /** The window attention prior, `heads` x 64 x 64 halves in fragment order. */
  | { kind: 'prior'; offset: number; heads: number }
  /** Per-head f32 attention scales of a window block (the slot is `alignUp(4 * heads, 16)` bytes). */
  | { kind: 'window-scale'; offset: number; heads: number }
  /** Per-head f32 learned scales of a ViT block. */
  | { kind: 'vit-scale'; offset: number; heads: number }
  /** One f16 constant (`block70.layer0.blend_scale`, the unread ViT `layer3`). */
  | { kind: 'half-constant'; offset: number; value: 'blend-scale' | 'vit-layer3' };

/** Bytes a region occupies. */
export function regionByteLength(region: NRRegion): number {
  switch (region.kind) {
    case 'fp8':
      return region.k * region.n;
    case 'f16-matrix':
      // Whole 16x16 tiles of halves (the head's N = 4 occupies one tile column).
      return alignUp(region.k, 16) * alignUp(region.n, 16) * 2;
    case 'skip-scale':
      return region.count * 2;
    case 'prior':
      return region.heads * 8192;
    case 'window-scale':
      return alignUp(region.heads * 4, 16);
    case 'vit-scale':
      return region.heads * 4;
    case 'half-constant':
      return 2;
  }
}

/** One `blockN.layerM.parameter` record of the manifest. */
export interface NRRecordLayout {
  readonly name: string;
  readonly block: number;
  readonly layer: number;
  readonly parameter: string;
  /** The record's length in the model directory (the synthetic generator writes exactly this). */
  readonly byteLength: number;
  /**
   * True where the reference graph insists on the exact length (block 0, the first decoder blocks, block 70);
   * otherwise a model is accepted when its record holds at least every region (`readEnd`).
   */
  readonly exact: boolean;
  /** End of the last region: the bytes the graph reads. */
  readonly readEnd: number;
  /** Regions in offset order. Bytes outside every region are padding (zero in the synthetic model). */
  readonly regions: readonly NRRegion[];
}

export const recordName = (block: number, layer = 0, parameter = 'layer'): string =>
  `block${block}.layer${layer}.${parameter}`;

/** Channels of the window blocks (`graph.js` `record`); 0 for the split, ViT and block 39 records. */
export function blockChannels(block: number): number {
  if (block <= 4 || (block >= 66 && block <= 70)) return 32;
  if (block <= 8 || (block >= 62 && block <= 65)) return 64;
  if (block <= 14 || (block >= 56 && block <= 61)) return 128;
  if (block <= 22 || (block >= 48 && block <= 55)) return 256;
  return 0;
}

/** The first block of each decoder stage (`upsampleFusedLayout`). */
export const DECODER_FIRST_BLOCKS: readonly number[] = [48, 56, 62, 66];
/** The last block of each encoder stage, whose record carries the C -> 2C transition. */
export const ENCODER_LAST_BLOCKS: readonly number[] = [4, 8, 14, 22];

const fp8 = (role: Fp8Role, offset: number, k: number, n: number, batchK = k): NRRegion => ({
  kind: 'fp8',
  role,
  offset,
  k,
  n,
  batchK,
});
const skip = (offset: number, count: number): NRRegion => ({ kind: 'skip-scale', offset, count });

/** The regions of one window block (`graph.js` `block`) from its layout. */
function windowBlockRegions(layout: FusedLayout, channels: number): NRRegion[] {
  const regions: NRRegion[] = [];
  if (layout.expertFfn) {
    const experts = layout.expertCount;
    const w2Base = layout.expand + experts * channels * 128;
    const w3Base = w2Base + experts * 128 * 32;
    regions.push(fp8('expand', layout.expand, experts * channels, 128, channels));
    regions.push(fp8('contract', w2Base, experts * 128, 32, 128));
    regions.push(fp8('merge', w3Base, channels, channels));
  } else {
    regions.push(fp8('expand', layout.expand, channels, layout.hidden));
    regions.push(fp8('contract', layout.contractWeights, layout.hidden, channels));
  }
  if (layout.inputAdapter !== undefined) {
    regions.push({ kind: 'f16-matrix', role: 'adapter', offset: layout.inputAdapter, k: 16, n: 32 });
  }
  if (layout.upsampleWeight !== undefined) {
    regions.push(fp8('transition', layout.upsampleWeight, channels * 2, channels));
  }
  regions.push(skip(layout.ffnCosSkip, channels));
  if (layout.transitionScale !== undefined) regions.push(skip(layout.transitionScale, channels));
  if (layout.inputScale !== undefined) regions.push(skip(layout.inputScale, 32));
  if (layout.adapterScale !== undefined) regions.push(skip(layout.adapterScale, 32));
  regions.push(fp8('qkv', layout.qkv, channels, channels * 3));
  regions.push({ kind: 'prior', offset: layout.relative, heads: layout.heads });
  regions.push({ kind: 'window-scale', offset: layout.scale, heads: layout.heads });
  regions.push(fp8('projection', layout.projection, channels, channels));
  regions.push(skip(layout.attnCosSkip, channels));
  if (layout.postWeights !== undefined) {
    regions.push({ kind: 'f16-matrix', role: 'head', offset: layout.postWeights, k: 32, n: 4 });
  }
  return regions;
}

function record(
  block: number,
  layer: number,
  parameter: string,
  byteLength: number,
  exact: boolean,
  regions: NRRegion[],
): NRRecordLayout {
  // oxlint-disable-next-line unicorn/no-array-sort -- sorts a fresh copy
  const sorted = [...regions].sort((a, b) => a.offset - b.offset);
  const readEnd = sorted.reduce((end, region) => Math.max(end, region.offset + regionByteLength(region)), 0);
  return {
    name: recordName(block, layer, parameter),
    block,
    layer,
    parameter,
    byteLength,
    exact,
    readEnd,
    regions: sorted,
  };
}

/** The four records of a 512-channel split block (`graph.js` `splitBlock`). */
function splitRecords(block: number): NRRecordLayout[] {
  const channels = 512;
  const branches = 8;
  const branchChannels = 64;
  const middleChannels = 256;
  const heads = 16;
  const w2Base = branches * channels * branchChannels;
  const w3Base = w2Base + branches * branchChannels * middleChannels;
  const qkvRelative = channels * channels * 3;
  const qkvScale = qkvRelative + heads * 8192;
  return [
    record(block, 0, 'layer', w3Base + branches * middleChannels * branchChannels, false, [
      fp8('expand', 0, channels, channels),
      fp8('expand', w2Base, branches * branchChannels, middleChannels, branchChannels),
      fp8('contract', w3Base, branches * middleChannels, branchChannels, middleChannels),
    ]),
    record(block, 1, 'layer', channels * channels + channels * 2, false, [
      fp8('merge', 0, channels, channels),
      skip(channels * channels, channels),
    ]),
    record(block, 2, 'layer', qkvScale + heads * 4, false, [
      fp8('qkv', 0, channels, channels * 3),
      { kind: 'prior', offset: qkvRelative, heads },
      { kind: 'window-scale', offset: qkvScale, heads },
    ]),
    record(block, 3, 'layer', channels * channels + channels * 2, false, [
      fp8('projection', 0, channels, channels),
      skip(channels * channels, channels),
    ]),
  ];
}

/** The five records of a ViT block (`graph.js` `vit`); the qkv record puts its head scales first. */
function vitRecords(block: number): NRRecordLayout[] {
  const channels = 1024;
  const heads = 32;
  const ffn = 4096;
  return [
    record(block, 0, 'layer', channels * ffn, false, [fp8('expand', 0, channels, ffn)]),
    record(block, 1, 'layer', ffn * channels + channels * 2, false, [
      fp8('contract', 0, ffn, channels),
      skip(ffn * channels, channels),
    ]),
    record(block, 2, 'layer', heads * 4 + channels * channels * 3, false, [
      { kind: 'vit-scale', offset: 0, heads },
      fp8('qkv', heads * 4, channels, channels * 3),
    ]),
    record(block, 3, 'layer', 2, false, [{ kind: 'half-constant', offset: 0, value: 'vit-layer3' }]),
    record(block, 4, 'layer', channels * channels + channels * 2, false, [
      fp8('projection', 0, channels, channels),
      skip(channels * channels, channels),
    ]),
  ];
}

function windowRecord(block: number): NRRecordLayout {
  const channels = blockChannels(block);
  if (block === 0) {
    const layout = preFusedLayout();
    return record(0, 0, 'layer', layout.endWithoutPadding + 16, true, windowBlockRegions(layout, 32));
  }
  if (block === 70) {
    const layout = postFusedLayout();
    return record(70, 0, 'layer', layout.endWithoutPadding, true, windowBlockRegions(layout, 32));
  }
  if (DECODER_FIRST_BLOCKS.includes(block)) {
    const layout = upsampleFusedLayout(channels * 2, channels);
    return record(block, 0, 'layer', layout.endWithoutPadding + 16, true, windowBlockRegions(layout, channels));
  }
  const layout = fusedLayout(channels);
  const regions = windowBlockRegions(layout, channels);
  if (ENCODER_LAST_BLOCKS.includes(block)) {
    // The next stage's C -> 2C input projection is appended in place of the trailing pad (graph.js:411-413, 445-449).
    regions.push(fp8('transition', layout.endWithoutPadding, channels, channels * 2));
    return record(block, 0, 'layer', layout.endWithoutPadding + channels * channels * 2, false, regions);
  }
  return record(block, 0, 'layer', layout.endWithoutPadding + 16, false, regions);
}

let cached: readonly NRRecordLayout[] | undefined;

/** All 153 records in block, layer order (the order the synthetic manifest lists them in). */
export function modelRecords(): readonly NRRecordLayout[] {
  if (cached) return cached;
  const records: NRRecordLayout[] = [];
  for (let block = 0; block < NR_BLOCK_COUNT; ++block) {
    if ((block >= 23 && block <= 30) || (block >= 40 && block <= 47)) {
      records.push(...splitRecords(block));
      if (block === 30) {
        // The 512 -> 1024 projection into the ViT (graph.js:476).
        records.push(record(30, 4, 'layer', 512 * 1024, false, [fp8('transition', 0, 512, 1024)]));
      }
    } else if (block >= 31 && block <= 38) {
      records.push(...vitRecords(block));
    } else if (block === 39) {
      // The 1024 -> 512 projection out of the ViT and the merge's skip scale (graph.js:482-490).
      records.push(
        record(39, 0, 'layer', 1024 * 512 + 512 * 2, false, [fp8('transition', 0, 1024, 512), skip(1024 * 512, 512)]),
      );
    } else {
      records.push(windowRecord(block));
      if (block === 70) {
        records.push(
          record(70, 0, 'blend_scale', 2, false, [{ kind: 'half-constant', offset: 0, value: 'blend-scale' }]),
        );
      }
    }
  }
  cached = records;
  return records;
}

/** The record layout by name. */
export function recordLayout(name: string): NRRecordLayout | undefined {
  return modelRecords().find((entry) => entry.name === name);
}
