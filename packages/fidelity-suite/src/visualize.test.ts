import { describe, expect, it } from 'vitest';

import { compareBytes } from './compare.js';
import { featuresFromFrame, proxyComponent } from './features.js';
import {
  blendWeight,
  boundaryCodes,
  e4Byte,
  enlarge,
  fitForDisplay,
  halve,
  headRgb,
  inputProxy,
  unitByte,
  type RgbImage,
} from './visualize.js';

const make = (width: number, height: number): RgbImage => ({
  width,
  height,
  data: new Uint8Array(width * height * 3).map((_, i) => (i * 37) & 0xff),
});

const geometry = { validWidth: 2, validHeight: 2, fullWidth: 3, fullHeight: 2 };

describe('e4Byte', () => {
  it('maps codes around 128 and is injective except for the two zeros', () => {
    expect([0x00, 0x01, 0x7e, 0x80, 0x81, 0xfe].map(e4Byte)).toEqual([128, 129, 254, 128, 127, 2]);
    const seen = new Map<number, number>();
    for (let code = 0; code < 256; ++code) {
      const byte = e4Byte(code);
      if (seen.has(byte)) expect([seen.get(byte), code]).toEqual([0x00, 0x80]);
      seen.set(byte, code);
    }
  });
});

describe('mappings', () => {
  it('inputProxy reads lanes 4-6 of the valid region as code values', () => {
    const features = new Float32Array(geometry.fullWidth * geometry.fullHeight * 16);
    features.set([-0.0625, 0, 0.0625], 4); // codes 0, 0.5, 1 at (0, 0)
    features.set([0.0625, 0.0625, 0.0625], (1 * 3 + 1) * 16 + 4); // (1, 1): white
    const image = inputProxy(features, geometry);
    expect([image.width, image.height]).toEqual([2, 2]);
    expect([...image.data.subarray(0, 3)]).toEqual([0, 128, 255]);
    expect([...image.data.subarray(9, 12)]).toEqual([255, 255, 255]);
  });

  it('headRgb and blendWeight are fixed functions of the head', () => {
    const head = new Float32Array(geometry.fullWidth * geometry.fullHeight * 4);
    head.set([2, -2, 0, 0], 0);
    head.set([0, 0, 0, Number.POSITIVE_INFINITY], 4);
    expect([...headRgb(head, geometry).data.subarray(0, 3)]).toEqual([255, 0, 128]);
    const blend = blendWeight(head, geometry, 0.5);
    expect([...blend.data.subarray(0, 6)]).toEqual([64, 64, 64, 128, 128, 128]);
    expect(unitByte(Number.NaN)).toBe(0);
  });

  it('boundaryCodes shows the first three channels and checks the shape', () => {
    const codes = Uint8Array.from([1, 0x81, 0, 9, 2, 0x82, 0x80, 9]);
    expect([...boundaryCodes(codes, 2, 1, 4).data]).toEqual([129, 127, 128, 130, 126, 128]);
    expect(() => boundaryCodes(codes, 3, 1, 4)).toThrow(RangeError);
  });

  it('fits images to 128-256 px by box halving and integer enlarging', () => {
    expect([fitForDisplay(make(576, 512)).width, fitForDisplay(make(576, 512)).height]).toEqual([144, 128]);
    expect(fitForDisplay(make(8, 8)).width).toBe(256);
    expect(fitForDisplay(make(12, 8)).width).toBe(264);
    expect(fitForDisplay(make(256, 256)).width).toBe(256);
    const small = { width: 2, height: 2, data: Uint8Array.from([0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4]) };
    expect([...halve(small).data]).toEqual([2, 2, 2]); // (0 + 1 + 2 + 4 + 2) >> 2
    expect([...enlarge(halve(small), 2).data]).toEqual(Array.from({ length: 12 }, () => 2));
  });
});

describe('compareBytes', () => {
  it('reports bit-exact, the first difference and length differences', () => {
    expect(compareBytes(Uint8Array.of(1, 2), Uint8Array.of(1, 2)).verdict).toBe('bit-exact');
    expect(compareBytes(Uint8Array.of(1, 3, 4), Uint8Array.of(1, 2, 5))).toEqual({
      verdict: 'differs',
      bytes: 3,
      mismatches: 2,
      first: 1,
    });
    expect(compareBytes(Uint8Array.of(1), Uint8Array.of(1, 2))).toMatchObject({ verdict: 'differs', first: 1 });
  });
});

describe('featuresFromFrame', () => {
  it('builds the proxy lanes, mirrors the padding and repeats the frame as history', () => {
    expect(proxyComponent(0)).toBe(0);
    expect(proxyComponent(Number.NaN)).toBe(0);
    expect(proxyComponent(-1)).toBe(0);
    expect(proxyComponent(1e9)).toBe(0); // not a finite half
    expect(proxyComponent(0.5)).toBeGreaterThan(proxyComponent(0.25));
    expect(proxyComponent(4)).toBeLessThanOrEqual(1);
    const linear = Float32Array.from([0, 0, 0, 1, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1]);
    const features = featuresFromFrame(linear, geometry);
    expect(features.length).toBe(3 * 2 * 16);
    const lane = (x: number, y: number, l: number) => features[(y * 3 + x) * 16 + l];
    expect(lane(0, 0, 4)).toBe(-0.0625);
    expect(lane(1, 0, 4)).toBeGreaterThan(-0.0625);
    expect(lane(2, 0, 4)).toBe(lane(0, 0, 4)); // mirrored: 2 * 2 - 2 - 2 = 0
    for (let l = 0; l < 3; ++l) expect(lane(1, 1, 7 + l)).toBe(lane(1, 1, 4 + l));
    expect([lane(0, 0, 3), lane(0, 0, 11), lane(0, 0, 13)]).toEqual([1, 1, -1]);
    expect(featuresFromFrame(linear, geometry)).toEqual(features);
  });
});
