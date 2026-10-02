// The 153 records of the model directory (OpenDLSS-NR's docs/weights.md, by maan, MIT) and their byte lengths.

import { describe, expect, it } from 'vitest';

import * as ref from '@ref/geometry.js';

import { blockChannels, modelRecords, recordLayout, regionByteLength, type NRRecordLayout } from './layouts.js';

const length = (name: string): number => recordLayout(name)!.byteLength;

const count = (predicate: (record: NRRecordLayout) => boolean) => modelRecords().filter(predicate).length;

describe('model records', () => {
  const records = modelRecords();

  it('are 153 uniquely named records over 71 blocks', () => {
    expect(records).toHaveLength(153);
    expect(new Set(records.map((record) => record.name)).size).toBe(153);
    expect(new Set(records.map((record) => record.block)).size).toBe(71);
    for (const record of records)
      expect(record.name).toBe(`block${record.block}.layer${record.layer}.${record.parameter}`);
  });

  it('group as 23 + 33 + 40 + 1 + 32 + 22 + 2', () => {
    expect(count((r) => r.block <= 22)).toBe(23);
    expect(count((r) => r.block >= 23 && r.block <= 30)).toBe(33);
    expect(count((r) => r.block >= 31 && r.block <= 38)).toBe(40);
    expect(count((r) => r.block === 39)).toBe(1);
    expect(count((r) => r.block >= 40 && r.block <= 47)).toBe(32);
    expect(count((r) => r.block >= 48 && r.block <= 69)).toBe(22);
    expect(count((r) => r.block === 70)).toBe(2);
  });

  it('have the byte lengths the reference layouts give', () => {
    // Plain window blocks: fusedLayout(C).endWithoutPadding + 16.
    for (const [block, channels] of [
      [1, 32],
      [5, 64],
      [9, 128],
      [15, 256],
      [49, 256],
      [57, 128],
      [63, 64],
      [67, 32],
    ]) {
      expect(length(`block${block}.layer0.layer`)).toBe(ref.fusedLayout(channels).endWithoutPadding + 16);
    }
    expect([1, 5, 9, 15].map((b) => length(`block${b}.layer0.layer`))).toEqual([20672, 61760, 197184, 689232]);
    // Last encoder blocks: the C -> 2C transition appended.
    expect([4, 8, 14, 22].map((b) => length(`block${b}.layer0.layer`))).toEqual([22704, 69936, 229936, 820288]);
    // First decoder blocks: the graph checks upsampleFusedLayout(2C, C).endWithoutPadding + 16 (graph.js:518).
    const firsts = [
      [66, 32],
      [62, 64],
      [56, 128],
      [48, 256],
    ];
    for (const [block, channels] of firsts) {
      expect(length(`block${block}.layer0.layer`)).toBe(
        ref.upsampleFusedLayout(channels * 2, channels).endWithoutPadding + 16,
      );
    }
    expect(firsts.map(([b]) => length(`block${b}.layer0.layer`))).toEqual([22784, 70048, 230176, 820784]);
    // Block 0 and block 70 (graph.js:365, 558).
    expect(length('block0.layer0.layer')).toBe(ref.preFusedLayout().endWithoutPadding + 16);
    expect(length('block0.layer0.layer')).toBe(21696);
    expect(length('block70.layer0.layer')).toBe(ref.postFusedLayout().endWithoutPadding);
    expect(length('block70.layer0.layer')).toBe(21808);
    expect(length('block70.layer0.blend_scale')).toBe(2);
    // Split, ViT and the two ViT transitions (graph.js:231-326, 476, 482-490).
    for (const block of [23, 30, 40, 47]) {
      expect([0, 1, 2, 3].map((layer) => length(`block${block}.layer${layer}.layer`))).toEqual([
        524288, 263168, 917568, 263168,
      ]);
    }
    expect(length('block30.layer4.layer')).toBe(524288);
    for (const block of [31, 38]) {
      expect([0, 1, 2, 3, 4].map((layer) => length(`block${block}.layer${layer}.layer`))).toEqual([
        4194304,
        4096 * 1024 + 2048,
        128 + 1024 * 3072,
        2,
        1024 * 1024 + 2048,
      ]);
    }
    expect(length('block39.layer0.layer')).toBe(1024 * 512 + 1024);
  });

  it('sum to the pinned total', () => {
    const total = records.reduce((sum, record) => sum + record.byteLength, 0);
    expect(total).toMatchInlineSnapshot(`147683618`);
  });

  it('hold non-overlapping regions inside each record', () => {
    for (const record of records) {
      let end = 0;
      for (const region of record.regions) {
        expect(region.offset, record.name).toBeGreaterThanOrEqual(end);
        end = region.offset + regionByteLength(region);
      }
      expect(end, record.name).toBe(record.readEnd);
      expect(record.readEnd, record.name).toBeLessThanOrEqual(record.byteLength);
      // What is left is at most the 16-byte trailing pad.
      expect(record.byteLength - record.readEnd, record.name).toBeLessThanOrEqual(16);
    }
  });

  it('give every window block its channel count', () => {
    expect([0, 4, 5, 8, 9, 14, 15, 22, 48, 55, 56, 61, 62, 65, 66, 70].map(blockChannels)).toEqual([
      32, 32, 64, 64, 128, 128, 256, 256, 256, 256, 128, 128, 64, 64, 32, 32,
    ]);
    expect([23, 31, 39, 47].map(blockChannels)).toEqual([0, 0, 0, 0]);
  });
});
