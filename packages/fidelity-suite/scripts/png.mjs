// A small deterministic PNG encoder (8-bit RGB, no metadata): equal pixels always give equal file bytes.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). No dependencies: node:zlib's deflate at a
// fixed level, and a fixed per-row filter choice (the one with the smallest sum of absolute residuals, ties to the
// lowest filter number), so the bytes depend on the pixels and the Node zlib build only.

import { crc32, deflateSync } from 'node:zlib';

const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'latin1');
  Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

const paeth = (a, b, c) => {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
};

/** Filter one row with filter `type` (0-4) into `out`. */
function filterRow(type, row, previous, out, bpp) {
  for (let i = 0; i < row.length; ++i) {
    const a = i >= bpp ? row[i - bpp] : 0;
    const b = previous ? previous[i] : 0;
    const c = previous && i >= bpp ? previous[i - bpp] : 0;
    const predicted = type === 0 ? 0 : type === 1 ? a : type === 2 ? b : type === 3 ? (a + b) >> 1 : paeth(a, b, c);
    out[i] = (row[i] - predicted) & 0xff;
  }
}

/** Encode `rgb` (`width * height * 3` bytes, rows top to bottom) as a PNG. */
export function encodePng(rgb, width, height) {
  if (rgb.length !== width * height * 3) throw new RangeError(`png: ${rgb.length} bytes for ${width}x${height} RGB`);
  const stride = width * 3;
  const raw = new Uint8Array((stride + 1) * height);
  const candidate = new Uint8Array(stride);
  for (let y = 0; y < height; ++y) {
    const row = rgb.subarray(y * stride, (y + 1) * stride);
    const previous = y ? rgb.subarray((y - 1) * stride, y * stride) : null;
    let best = 0;
    let bestCost = Infinity;
    for (let type = 0; type < 5; ++type) {
      filterRow(type, row, previous, candidate, 3);
      let cost = 0;
      for (const value of candidate) cost += value < 128 ? value : 256 - value;
      if (cost < bestCost) {
        bestCost = cost;
        best = type;
      }
    }
    raw[y * (stride + 1)] = best;
    filterRow(best, row, previous, raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), 3);
  }
  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header.set([8, 2, 0, 0, 0], 8); // 8-bit, truecolour, deflate, adaptive filtering, no interlace
  return Buffer.concat([
    SIGNATURE,
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw, { level: 9, memLevel: 9 })),
    chunk('IEND', new Uint8Array(0)),
  ]);
}
