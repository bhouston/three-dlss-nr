// Deterministic input features for the browser harness (shim parity check and backend benchmark).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). The lane layout follows the design's
// synthetic features (section 5.1; OpenDLSS-NR docs/frame.md): 0-2 noise, 3 = 1, 4-6 the centred proxy, 7-9 a shifted
// copy as history, 10 = 0, 11 = 1, 12 = 1, 13 = -1, 14 = -1, 15 = 0. Values are on the half grid.
// TODO(chunk A): replace with `syntheticFeatures` from src/synthetic/features.ts once it lands; any deterministic
// input serves this harness, since both sides of every comparison get the same bytes.

/** Round to the nearest half (ties to even). */
export const toHalf = (value: number): number =>
  (Math as unknown as { f16round: (x: number) => number }).f16round(value);

/** The value of the half bit pattern `bits`. */
export function halfToNumber(bits: number): number {
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  const sign = bits & 0x8000 ? -1 : 1;
  if (exponent === 0x1f) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24);
}

/** The half bit pattern of `value`, rounded to nearest even (NaN -> 0x7e00). */
export function halfBits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  const rounded = toHalf(value);
  const sign = rounded < 0 || Object.is(rounded, -0) ? 0x8000 : 0;
  const magnitude = Math.abs(rounded);
  if (magnitude === Infinity) return sign | 0x7c00;
  if (magnitude < 2 ** -14) return sign | Math.round(magnitude * 2 ** 24);
  const exponent = Math.floor(Math.log2(magnitude));
  const e = magnitude >= 2 ** (exponent + 1) ? exponent + 1 : magnitude < 2 ** exponent ? exponent - 1 : exponent;
  return sign | ((e + 15) << 10) | Math.round((magnitude / 2 ** e - 1) * 1024);
}

/** The centred proxy code value of docs/frame.md. */
const centre = (value: number) => toHalf(toHalf(toHalf(value) - 0.5) * 0.125);

function xorshift(seed: number): () => number {
  let state = seed >>> 0 || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** `f32 [fullWidth * fullHeight][16]`. */
export function testFeatures(fullWidth: number, fullHeight: number, seed = 1): Float32Array {
  const random = xorshift(seed);
  const proxy = (x: number, y: number, c: number) => {
    const checker = ((x >> 4) + (y >> 4)) & 1;
    const gradient = (x / fullWidth + y / fullHeight + c * 0.25) % 1;
    return 0.6 * gradient + 0.3 * checker + 0.05 * c;
  };
  const out = new Float32Array(fullWidth * fullHeight * 16);
  for (let y = 0; y < fullHeight; ++y) {
    for (let x = 0; x < fullWidth; ++x) {
      const base = (y * fullWidth + x) * 16;
      const gauss = () => (random() + random() + random() + random() - 2) * 1.7320508;
      out[base] = toHalf(gauss());
      out[base + 1] = toHalf(gauss());
      out[base + 2] = toHalf(gauss());
      out[base + 3] = 1;
      for (let c = 0; c < 3; ++c) {
        out[base + 4 + c] = centre(proxy(x, y, c));
        out[base + 7 + c] = centre(proxy(x + 2, y + 1, c));
      }
      out[base + 10] = 0;
      out[base + 11] = 1;
      out[base + 12] = 1;
      out[base + 13] = -1;
      out[base + 14] = -1;
      out[base + 15] = 0;
    }
  }
  return out;
}
