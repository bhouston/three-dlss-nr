// Deterministic synthetic weights in the reference's model directory layout.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). NVIDIA's weights are proprietary and are not part of
// this project; these synthetic weights stand in for them in tests and in the demo. They follow the reference's
// docs/weights.md layout exactly (manifest.json + model/stages/*.bin, 153 records), so the reference WebGPU port loads
// them unchanged, and the TSL port can be compared with it byte for byte.
//
// Determinism: only integer arithmetic, IEEE double + - * / and one correctly rounded sqrt per matrix shape - no
// transcendentals - so every engine on every OS produces the same bytes (the stage hashes are pinned in the tests).
// The PRNG is the reference's 32-bit xorshift (src/numerics_cases.js), seeded per record by
// `fnv1a32(name) ^ seed`, so changing one record never shifts another.

import { alignUp } from '../geometry.js';
import {
  modelRecords,
  NR_BLOCK_COUNT,
  regionByteLength,
  type NRRecordLayout,
  type NRRegion,
} from '../model/layouts.js';
import { sha256Hex, type NRManifest, type NRManifestStage, type NRManifestTensor } from '../model/manifest.js';
import { e4m3FromNumber, f16Bits, packedF16WeightIndex } from '../numerics/oracle.js';
import {
  SYNTHETIC_CONSTANTS,
  SYNTHETIC_F16_SIGMA,
  SYNTHETIC_GAINS,
  SYNTHETIC_RANGES,
  SYNTHETIC_ZERO_THRESHOLD,
} from './gains.js';

export const SYNTHETIC_FORMAT = 'three-dlss-nr synthetic v1';

/** Stage files of the synthetic model: `[id, first block, last block]`, grouped by level. */
export const SYNTHETIC_STAGES: readonly (readonly [string, number, number])[] = [
  ['s00', 0, 4], // field + encoder 32
  ['s01', 5, 8], // encoder 64
  ['s02', 9, 14], // encoder 128
  ['s03', 15, 22], // encoder 256
  ['s04', 23, 30], // encoder 512 + the projection into the ViT
  ['s05', 31, 34], // ViT
  ['s06', 35, 38], // ViT
  ['s07', 39, 47], // decoder 512
  ['s08', 48, 55], // decoder 256
  ['s09', 56, 61], // decoder 128
  ['s10', 62, 70], // decoder 64, 32, post block + head
];

/** 32-bit FNV-1a of a string's UTF-16 code units (ASCII names). */
export function fnv1a32(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; ++i) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return hash >>> 0;
}

/** The xorshift32 state of a record (never 0, which xorshift cannot leave). */
export const recordSeed = (name: string, seed: number): number => (fnv1a32(name) ^ seed) >>> 0 || 0x9e3779b9;

/**
 * Approximate standard normal from one 32-bit draw: Irwin-Hall over its four bytes (sum of four uniforms on
 * 0..255, mean 510, variance 4 * (256^2 - 1) / 12 = 21845), centred and scaled.
 */
const IRWIN_HALL_MEAN = 510;
const IRWIN_HALL_SIGMA = Math.sqrt(21845);
const byteSum = (draw: number): number =>
  (draw & 0xff) + ((draw >>> 8) & 0xff) + ((draw >>> 16) & 0xff) + (draw >>> 24);
export const irwinHallNormal = (draw: number): number => (byteSum(draw) - IRWIN_HALL_MEAN) / IRWIN_HALL_SIGMA;

/** Uniform in `[low, high)` from one draw (exact in double before the final rounding). */
const uniform = (draw: number, [low, high]: readonly [number, number]): number =>
  low + (high - low) * (draw / 4294967296);

/**
 * E4M3 code of every Irwin-Hall byte sum (0..1020) for one gain and fan-in: `e4m3(g * N / sqrt(fanIn))`, magnitude
 * clamped to 0x51 (|w| <= 9, the reference's load-time bound) and never the NaN code.
 */
function fp8CodeTable(gain: number, fanIn: number): Uint8Array {
  const table = new Uint8Array(1021);
  const scale = gain / Math.sqrt(fanIn);
  for (let sum = 0; sum <= 1020; ++sum) {
    let code = e4m3FromNumber(((sum - IRWIN_HALL_MEAN) / IRWIN_HALL_SIGMA) * scale);
    if ((code & 0x7f) > 0x51) code = (code & 0x80) | 0x51;
    table[sum] = code;
  }
  return table;
}

const tables = new Map<string, Uint8Array>();
const codeTable = (gain: number, fanIn: number): Uint8Array => {
  const key = `${gain}/${fanIn}`;
  let table = tables.get(key);
  if (!table) tables.set(key, (table = fp8CodeTable(gain, fanIn)));
  return table;
};

/** The skip-scale range of a block: the ViT's (31-38), the decoder window blocks' (48-70), or the general one. */
export function skipScaleRange(block: number): readonly [number, number] {
  if (block >= 31 && block <= 38) return SYNTHETIC_RANGES.vitSkipScale;
  if (block >= 48) return SYNTHETIC_RANGES.decoderSkipScale;
  return SYNTHETIC_RANGES.skipScale;
}

/** Fill `region` of `bytes` (a record) from the xorshift state `state`; returns the new state. */
function fillRegion(bytes: Uint8Array, region: NRRegion, state: number, block: number): number {
  let x = state;
  const next = (): number => {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x;
  };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const putHalf = (at: number, value: number) => view.setUint16(at, f16Bits(value), true);
  switch (region.kind) {
    case 'fp8': {
      // Fragment order is irrelevant for iid values: the bytes are filled in storage order.
      const table = codeTable(SYNTHETIC_GAINS[region.role], region.batchK);
      const end = region.offset + region.k * region.n;
      for (let at = region.offset; at < end; ++at) {
        // Two draws per weight, always: one decides the forced zero, one is the value.
        x = (x ^ (x << 13)) >>> 0;
        x = (x ^ (x >>> 17)) >>> 0;
        x = (x ^ (x << 5)) >>> 0;
        const zero = x < SYNTHETIC_ZERO_THRESHOLD;
        x = (x ^ (x << 13)) >>> 0;
        x = (x ^ (x >>> 17)) >>> 0;
        x = (x ^ (x << 5)) >>> 0;
        bytes[at] = zero ? 0 : table[byteSum(x)];
      }
      break;
    }
    case 'f16-matrix': {
      const sigma = SYNTHETIC_F16_SIGMA[region.role];
      for (let k = 0; k < region.k; ++k) {
        for (let n = 0; n < region.n; ++n) {
          putHalf(region.offset + packedF16WeightIndex(k, n, region.n) * 2, irwinHallNormal(next()) * sigma);
        }
      }
      break;
    }
    case 'skip-scale':
      for (let i = 0; i < region.count; ++i) putHalf(region.offset + i * 2, uniform(next(), skipScaleRange(block)));
      break;
    case 'prior':
      for (let i = 0; i < region.heads * 4096; ++i)
        putHalf(region.offset + i * 2, uniform(next(), SYNTHETIC_RANGES.prior));
      break;
    case 'window-scale':
    case 'vit-scale': {
      const range = region.kind === 'window-scale' ? SYNTHETIC_RANGES.windowScale : SYNTHETIC_RANGES.vitScale;
      for (let h = 0; h < region.heads; ++h) view.setFloat32(region.offset + h * 4, uniform(next(), range), true);
      break;
    }
    case 'half-constant':
      putHalf(region.offset, SYNTHETIC_CONSTANTS[region.value]);
      break;
  }
  return x;
}

/** The bytes of one record (zero outside its regions). */
export function generateSyntheticRecord(record: NRRecordLayout, seed = 1): Uint8Array {
  const bytes = new Uint8Array(record.byteLength);
  let state = recordSeed(record.name, seed);
  for (const region of record.regions) {
    if (region.offset + regionByteLength(region) > record.byteLength) {
      throw new Error(`region at ${region.offset} exceeds ${record.name}`);
    }
    state = fillRegion(bytes, region, state, record.block);
  }
  return bytes;
}

/** A model directory in memory: `manifest.json` and `model/stages/sNN.bin`. */
export interface SyntheticModel {
  readonly manifest: NRManifest;
  /** Paths relative to the model directory (`manifest.json`, `model/stages/s00.bin`, ...). */
  readonly files: Map<string, Uint8Array>;
}

export interface SyntheticModelOptions {
  /** PRNG seed (default 1). */
  seed?: number;
}

/**
 * Generate the whole synthetic model (~141 MiB) in memory. Feed `files` to `NRModel.load({ files })`, to the
 * reference through the test fetch shim (`registerSyntheticFiles`), or write it to disk
 * (`scripts/make-synthetic-model.mjs`).
 */
export async function generateSyntheticModel({ seed = 1 }: SyntheticModelOptions = {}): Promise<SyntheticModel> {
  const records = modelRecords();
  const files = new Map<string, Uint8Array>();
  const stages: NRManifestStage[] = [];
  const tensors: NRManifestTensor[] = [];
  for (const [id, first, last] of SYNTHETIC_STAGES) {
    const members = records.filter((record) => record.block >= first && record.block <= last);
    const offsets: number[] = [];
    let length = 0;
    for (const record of members) {
      length = alignUp(length, 16);
      offsets.push(length);
      length += record.byteLength;
    }
    const stage = new Uint8Array(length);
    members.forEach((record, index) => {
      stage.set(generateSyntheticRecord(record, seed), offsets[index]);
      tensors.push({
        name: record.name,
        block: record.block,
        layer: record.layer,
        stage: id,
        stageOffset: offsets[index],
        byteLength: record.byteLength,
      });
    });
    const file = `stages/${id}.bin`;
    files.set(`model/${file}`, stage);
    stages.push({ id, file, packedByteLength: length, sha256: await sha256Hex(stage) });
  }
  const manifest: NRManifest = {
    format: SYNTHETIC_FORMAT,
    seed,
    totals: { blockCount: NR_BLOCK_COUNT },
    stages,
    tensors,
  };
  files.set('manifest.json', new TextEncoder().encode(`${JSON.stringify(manifest, null, 2)}\n`));
  return { manifest, files };
}
