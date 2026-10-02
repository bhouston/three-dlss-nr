// The synthetic model: deterministic bytes (pinned hashes), the reference's directory layout, bounded weights.

import { beforeAll, describe, expect, it } from 'vitest';

import { manifestProblems, sha256Hex } from '../model/manifest.js';
import { firstUnboundedWeight } from '../model/Model.js';
import { modelRecords, regionByteLength } from '../model/layouts.js';
import { f16ToNumber, roundF16 } from '../numerics/oracle.js';
import {
  fnv1a32,
  generateSyntheticModel,
  generateSyntheticRecord,
  irwinHallNormal,
  skipScaleRange,
  SYNTHETIC_FORMAT,
  type SyntheticModel,
} from './generate.js';
import { SYNTHETIC_RANGES } from './gains.js';

let model: SyntheticModel;

beforeAll(async () => {
  model = await generateSyntheticModel({ seed: 1 });
});

const sameBytes = (a: Uint8Array, b: Uint8Array): boolean =>
  a.length === b.length &&
  Buffer.from(a.buffer, a.byteOffset, a.length).equals(Buffer.from(b.buffer, b.byteOffset, b.length));

const recordBytes = (name: string): Uint8Array => {
  const entry = model.manifest.tensors.find((tensor) => tensor.name === name)!;
  const stage = model.files.get(`model/stages/${entry.stage}.bin`)!;
  return stage.subarray(entry.stageOffset, entry.stageOffset + entry.byteLength);
};

/** Within bounds rounded to the stored precision (0.95 is 0.9502 as a half). */
const inRange = (value: number, [low, high]: readonly [number, number]) =>
  value >= roundF16(low) && value <= roundF16(high);

describe('synthetic model', () => {
  it('is a valid manifest of 11 stages and 153 records', () => {
    expect(model.manifest.format).toBe(SYNTHETIC_FORMAT);
    expect(model.manifest.totals.blockCount).toBe(71);
    expect(model.manifest.stages).toHaveLength(11);
    expect(model.manifest.tensors).toHaveLength(153);
    expect(manifestProblems(model.manifest)).toEqual([]);
    const parsed = JSON.parse(new TextDecoder().decode(model.files.get('manifest.json')));
    expect(parsed).toEqual(model.manifest);
    for (const tensor of model.manifest.tensors) expect(tensor.stageOffset % 16).toBe(0);
    expect(model.files.size).toBe(12);
  });

  it('writes every record at the length of its layout, and the byte sums agree', () => {
    const lengths = new Map(modelRecords().map((record) => [record.name, record.byteLength]));
    let sum = 0;
    for (const tensor of model.manifest.tensors) {
      expect(tensor.byteLength, tensor.name).toBe(lengths.get(tensor.name));
      sum += tensor.byteLength;
    }
    expect(sum).toBe(modelRecords().reduce((total, record) => total + record.byteLength, 0));
    let packed = 0;
    for (const stage of model.manifest.stages) {
      expect(model.files.get(`model/${stage.file}`)!.byteLength).toBe(stage.packedByteLength);
      packed += stage.packedByteLength;
    }
    // Stage files hold the records plus at most 15 bytes of alignment between consecutive records.
    expect(packed - sum).toBeGreaterThanOrEqual(0);
    expect(packed - sum).toBeLessThan(16 * 153);
  });

  it('has the pinned stage hashes (identical on every engine and OS)', async () => {
    const hashes = Object.fromEntries(model.manifest.stages.map((stage) => [stage.id, stage.sha256]));
    for (const stage of model.manifest.stages) {
      expect(await sha256Hex(model.files.get(`model/${stage.file}`)!)).toBe(stage.sha256);
    }
    expect(hashes).toMatchInlineSnapshot(`
      {
        "s00": "0dda0101a0eb47b05bd43419a351a887e4436a6322a8b6d048d091f4b8a9ee84",
        "s01": "abb2c1f2e1f29d1816e3af7f9a163d7801731d2f6016efd1205c76647d113029",
        "s02": "07f5b6a55339b0ba03b4ff27be6dbd203378a99ddd9d9646d8ea08f2e269ebb5",
        "s03": "3cffeaa7d6d8671c873daf7b753a17d4fa8e3795a83b064f291bd6bec6106887",
        "s04": "fdb975d9306defe141ee63c491a564b3455c20aa16b509c68251a64a99d38c25",
        "s05": "0027d19262d7c424f42e9efa60dff74371a7a7bb9e4b96e82f5e242302ba71fd",
        "s06": "27c84395c41f24af54bb58b189d679a62af63c851f1b1d13f318971cb9025022",
        "s07": "3647ea2b3dd8bbc2fc447fce6874548dd1400af6adc7eba78204693568a45607",
        "s08": "1ed53aa0f3f83cf806864c7f9fde767d8799bb13d0d6b8dc8969b8bbdc9504de",
        "s09": "cc9cbb0596eb6449b17f93708ac05af0d919f037f22c2b3be15bf36fe1c8a5ff",
        "s10": "3f57fb34c0ea5f7911553935999a8450fd7651b0a05c4c0a51782c2ddaf1acea",
      }
    `);
  });

  it('regenerates each record independently and identically', () => {
    for (const record of modelRecords().filter((r) => r.byteLength < 1 << 24)) {
      expect(sameBytes(generateSyntheticRecord(record, 1), recordBytes(record.name)), record.name).toBe(true);
    }
    const record = modelRecords()[1];
    expect(generateSyntheticRecord(record, 2)).not.toEqual(generateSyntheticRecord(record, 1));
  });

  it('keeps every FP8 weight within |w| <= 9, never NaN, with about 4% forced zeros', () => {
    let weights = 0;
    let zeros = 0;
    let signedZeros = 0;
    for (const record of modelRecords()) {
      const bytes = recordBytes(record.name);
      for (const region of record.regions) {
        if (region.kind !== 'fp8') continue;
        const matrix = bytes.subarray(region.offset, region.offset + region.k * region.n);
        expect(firstUnboundedWeight(matrix), record.name).toBe(-1);
        for (const code of matrix) {
          if ((code & 0x7f) === 0x7f) throw new Error(`NaN code in ${record.name}`);
          if (code === 0) zeros += 1;
          if (code === 0x80) signedZeros += 1;
        }
        weights += matrix.length;
      }
    }
    expect(weights).toBeGreaterThan(140e6);
    // Forced zeros (4 %) plus values that round to +0; negatives that round to zero keep their sign.
    expect(zeros / weights).toBeGreaterThan(0.04);
    expect((zeros + signedZeros) / weights).toBeLessThan(0.12);
    expect(signedZeros).toBeGreaterThan(0);
  });

  it('fills scales, priors and constants inside their ranges, and padding with zeros', () => {
    for (const record of modelRecords()) {
      const bytes = recordBytes(record.name);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const covered = new Uint8Array(bytes.length);
      for (const region of record.regions) {
        covered.fill(1, region.offset, region.offset + regionByteLength(region));
        const half = (i: number) => f16ToNumber(view.getUint16(region.offset + i * 2, true));
        switch (region.kind) {
          case 'skip-scale':
            for (let i = 0; i < region.count; ++i) expect(inRange(half(i), skipScaleRange(record.block))).toBe(true);
            break;
          case 'prior':
            for (let i = 0; i < region.heads * 4096; ++i) expect(inRange(half(i), SYNTHETIC_RANGES.prior)).toBe(true);
            break;
          case 'window-scale':
          case 'vit-scale': {
            const range = region.kind === 'window-scale' ? SYNTHETIC_RANGES.windowScale : SYNTHETIC_RANGES.vitScale;
            for (let h = 0; h < region.heads; ++h) {
              expect(inRange(view.getFloat32(region.offset + h * 4, true), range)).toBe(true);
            }
            if (region.kind === 'window-scale') {
              for (let at = region.offset + region.heads * 4; at < region.offset + regionByteLength(region); ++at) {
                expect(bytes[at]).toBe(0);
              }
            }
            break;
          }
          case 'half-constant':
            expect(view.getUint16(region.offset, true)).toBe(region.value === 'blend-scale' ? 0x3a00 : 0x3c00);
            break;
          default:
        }
      }
      for (let at = 0; at < bytes.length; ++at)
        if (!covered[at] && bytes[at] !== 0) throw new Error(`${record.name}@${at}`);
    }
  });

  it('draws from a seeded, integer-only PRNG', () => {
    expect(fnv1a32('')).toBe(0x811c9dc5);
    expect(fnv1a32('a')).toBe(0xe40c292c);
    expect(fnv1a32('foobar')).toBe(0xbf9cf968);
    expect(irwinHallNormal(0x7f7f8080)).toBe(0);
    expect(irwinHallNormal(0x00000000)).toBeCloseTo(-510 / Math.sqrt(21845), 15);
    expect(irwinHallNormal(0xffffffff)).toBeCloseTo(510 / Math.sqrt(21845), 15);
  });
});
