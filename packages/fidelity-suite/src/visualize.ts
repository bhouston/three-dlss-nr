// Deterministic tensor -> RGB8 mappings for the parity suite: equal tensors give equal pixels, so equal PNG bytes.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Every mapping runs on the CPU from read-back raw
// tensors, with the same code for every renderer. They are views for people; byte equality of the raw tensors is
// established separately (exactness.json), because a picture of three channels out of 32 cannot show everything.

/** An 8-bit RGB image, rows top to bottom. */
export interface RgbImage {
  width: number;
  height: number;
  /** `width * height * 3` bytes. */
  data: Uint8Array;
}

/** The padded field and the valid region of the network's input (the subset of `NRGeometry` used here). */
export interface FieldGeometry {
  validWidth: number;
  validHeight: number;
  fullWidth: number;
  fullHeight: number;
}

/** Visualizations wider than this are halved (2x2 box mean of the bytes) until they fit. */
export const MAX_WIDTH = 256;
/** Visualizations narrower than this are enlarged by an integer nearest-neighbour factor to at least `MAX_WIDTH`. */
export const MIN_WIDTH = 128;

const clamp01 = (value: number): number => (value > 1 ? 1 : value >= 0 ? value : 0);
/** `round(clamp(x, 0, 1) * 255)`; NaN maps to 0. */
export const unitByte = (value: number): number => Math.round(clamp01(value) * 255);

function image(width: number, height: number): RgbImage {
  return { width, height, data: new Uint8Array(width * height * 3) };
}

/**
 * `input`: the display proxy the network was given, from feature lanes 4-6 (`centred = (code - 0.5) / 8`, so
 * `code = lane * 8 + 0.5`), over the valid region.
 */
export function inputProxy(features: Float32Array, geometry: FieldGeometry): RgbImage {
  const { validWidth, validHeight, fullWidth } = geometry;
  const out = image(validWidth, validHeight);
  for (let y = 0; y < validHeight; ++y) {
    for (let x = 0; x < validWidth; ++x) {
      const source = (y * fullWidth + x) * 16;
      const target = (y * validWidth + x) * 3;
      for (let c = 0; c < 3; ++c) out.data[target + c] = unitByte(features[source + 4 + c] * 8 + 0.5);
    }
  }
  return out;
}

/** `head-rgb`: the head's RGB residual, `0.5 + rgb / 4`, over the valid region. */
export function headRgb(head: Float32Array, geometry: FieldGeometry): RgbImage {
  const { validWidth, validHeight, fullWidth } = geometry;
  const out = image(validWidth, validHeight);
  for (let y = 0; y < validHeight; ++y) {
    for (let x = 0; x < validWidth; ++x) {
      const source = (y * fullWidth + x) * 4;
      const target = (y * validWidth + x) * 3;
      for (let c = 0; c < 3; ++c) out.data[target + c] = unitByte(0.5 + head[source + c] / 4);
    }
  }
  return out;
}

/** `blend-logit`: the temporal blend weight `sigmoid(logit) * blendScale` as grey, over the valid region. */
export function blendWeight(head: Float32Array, geometry: FieldGeometry, blendScale: number): RgbImage {
  const { validWidth, validHeight, fullWidth } = geometry;
  const out = image(validWidth, validHeight);
  for (let y = 0; y < validHeight; ++y) {
    for (let x = 0; x < validWidth; ++x) {
      const logit = head[(y * fullWidth + x) * 4 + 3];
      const grey = unitByte((1 / (1 + Math.exp(-logit))) * blendScale);
      out.data.fill(grey, (y * validWidth + x) * 3, (y * validWidth + x) * 3 + 3);
    }
  }
  return out;
}

/** The RGB of an RGBA8 image (e.g. the reference's `composeImage`), alpha dropped. */
export function fromRgba(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): RgbImage {
  const out = image(width, height);
  for (let i = 0; i < width * height; ++i) {
    out.data[i * 3] = rgba[i * 4];
    out.data[i * 3 + 1] = rgba[i * 4 + 1];
    out.data[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return out;
}

/**
 * One E4M3 code as a byte: `128 + magnitude` for a positive code, `128 - magnitude` for a negative one (magnitude =
 * the low seven bits). Injective except that +0 and -0 both map to 128; exactness.json compares the raw codes.
 */
export const e4Byte = (code: number): number => (code & 0x80 ? 128 - (code & 0x7f) : 128 + (code & 0x7f));

/**
 * A block boundary: E4M3 codes `[width * height][channels]`, rows in row-major pixel order (the reference's layout
 * at every level); the pixel shows channels 0-2 through `e4Byte`.
 */
export function boundaryCodes(codes: Uint8Array, width: number, height: number, channels: number): RgbImage {
  if (codes.length !== width * height * channels) {
    throw new RangeError(`boundary: ${codes.length} bytes, expected ${width}x${height}x${channels}`);
  }
  const out = image(width, height);
  for (let pixel = 0; pixel < width * height; ++pixel) {
    for (let c = 0; c < 3; ++c) out.data[pixel * 3 + c] = e4Byte(codes[pixel * channels + c]);
  }
  return out;
}

/** Halve with a 2x2 box mean (`(a + b + c + d + 2) >> 2`); an odd last row or column is dropped. */
export function halve(source: RgbImage): RgbImage {
  const width = source.width >> 1;
  const height = source.height >> 1;
  const out = image(width, height);
  const at = (x: number, y: number, c: number) => source.data[(y * source.width + x) * 3 + c];
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      for (let c = 0; c < 3; ++c) {
        const sum =
          at(2 * x, 2 * y, c) + at(2 * x + 1, 2 * y, c) + at(2 * x, 2 * y + 1, c) + at(2 * x + 1, 2 * y + 1, c);
        out.data[(y * width + x) * 3 + c] = (sum + 2) >> 2;
      }
    }
  }
  return out;
}

/** Enlarge by an integer nearest-neighbour factor. */
export function enlarge(source: RgbImage, factor: number): RgbImage {
  const out = image(source.width * factor, source.height * factor);
  for (let y = 0; y < out.height; ++y) {
    for (let x = 0; x < out.width; ++x) {
      const from = (Math.floor(y / factor) * source.width + Math.floor(x / factor)) * 3;
      out.data.set(source.data.subarray(from, from + 3), (y * out.width + x) * 3);
    }
  }
  return out;
}

/** Bring a visualization to a viewable, committable width: halve while wider than 256, enlarge if under 128. */
export function fitForDisplay(source: RgbImage): RgbImage {
  let result = source;
  while (result.width > MAX_WIDTH) result = halve(result);
  if (result.width < MIN_WIDTH) result = enlarge(result, Math.ceil(MAX_WIDTH / result.width));
  return result;
}
