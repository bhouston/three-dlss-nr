import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { encodePng } from './png.mjs';

/** Undo the PNG row filters of an 8-bit RGB image. */
function decode(png: Uint8Array): { width: number; height: number; rgb: Uint8Array } {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  const idatLength = view.getUint32(33);
  const raw = inflateSync(png.subarray(41, 41 + idatLength));
  const stride = width * 3;
  const rgb = new Uint8Array(stride * height);
  for (let y = 0; y < height; ++y) {
    const type = raw[y * (stride + 1)];
    for (let i = 0; i < stride; ++i) {
      const a = i >= 3 ? rgb[y * stride + i - 3] : 0;
      const b = y ? rgb[(y - 1) * stride + i] : 0;
      const c = y && i >= 3 ? rgb[(y - 1) * stride + i - 3] : 0;
      const p = a + b - c;
      const paeth =
        Math.abs(p - a) <= Math.abs(p - b) && Math.abs(p - a) <= Math.abs(p - c)
          ? a
          : Math.abs(p - b) <= Math.abs(p - c)
            ? b
            : c;
      const predicted = [0, a, b, (a + b) >> 1, paeth][type];
      rgb[y * stride + i] = (raw[y * (stride + 1) + 1 + i] + predicted) & 0xff;
    }
  }
  return { width, height, rgb };
}

describe('encodePng', () => {
  const width = 7;
  const height = 5;
  const rgb = new Uint8Array(width * height * 3).map((_, i) => (i * 97 + (i >> 4) * 13) & 0xff);

  it('round-trips the pixels', () => {
    const decoded = decode(encodePng(rgb, width, height));
    expect([decoded.width, decoded.height]).toEqual([width, height]);
    expect(decoded.rgb).toEqual(rgb);
  });

  it('is deterministic and changes with any pixel', () => {
    const png = encodePng(rgb, width, height);
    expect(encodePng(rgb.slice(), width, height)).toEqual(png);
    const changed = rgb.slice();
    changed[40] ^= 1;
    expect(encodePng(changed, width, height)).not.toEqual(png);
    expect(() => encodePng(rgb, width + 1, height)).toThrow(RangeError);
  });
});
