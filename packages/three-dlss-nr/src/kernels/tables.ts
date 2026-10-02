// Lookup tables the FP8 GEMM reads, built on the CPU.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The reference builds its SiLU publication table on
// the GPU, once per device, with hardware half arithmetic (ports/browser-webgpu/src/matmul/silu-table.js
// `createSiluTable`, then packed-activation.js `createPackedSiluTable`): one E4M3 code per half bit pattern of the
// accumulator, `exact_e4_output_code(f16(fp8_domain(mp_cubic_silu(h))))`. We build the same 64 KiB on the CPU from
// the TS oracle (`siluE4CodeTable`, which is checked against the reference's numerics fixture for every half), so the
// table does not depend on the device's f16 rounding (lavapipe, for one, does not round `f32(f16(x))`).
//
// Signed zeros: a zero activation (+0 or -0) publishes 0x00 (`fp8_domain` returns +0 for a zero input); a nonzero
// activation that quantizes to zero keeps its sign (0x80 for a small negative one), unlike `publish_e4_code`.

import { e4m3ToNumber, f32Bits, siluE4CodeTable } from '../numerics/oracle.js';
import { attributeFromBytes } from '../tensors.js';
import type { StorageBufferAttribute } from '../types.js';

let siluCodes: Uint8Array | null = null;
let siluAttribute: StorageBufferAttribute | null = null;

/** The SiLU publication table: byte `h` is the E4M3 code the GEMM publishes for an accumulator with half bits `h`. */
export function siluCodeTable(): Uint8Array {
  siluCodes ??= siluE4CodeTable();
  return siluCodes;
}

/**
 * The SiLU table as a storage attribute (16384 words, byte `h` of the table in byte `h % 4` of word `h / 4`), shared
 * by every GEMM kernel: one attribute, so one GPU buffer per renderer.
 */
export function siluCodeTableAttribute(): StorageBufferAttribute {
  siluAttribute ??= attributeFromBytes(siluCodeTable());
  return siluAttribute;
}

/** Bias of the exponent in the low byte of an `e4OperandTable` entry. */
export const E4_OPERAND_EXPONENT_BIAS = 128;

/**
 * The FP8 GEMM's operand decode, one u32 per E4M3 code (the reference's `weight_metadata`, matmul/weight-table.js, in
 * our representation): bits 31-20 are the top of the f32 pattern of the decoded value (every E4M3 value, subnormals
 * included, has at most 4 significant bits, so its f32 pattern ends in 20 zero bits; the NaN codes decode to a signed
 * zero), and the low byte is `exponent + 128` with `exponent = max(field - 7, -6)`, or -100 for a zero or a NaN code
 * (excluded from every shared-exponent maximum, whose floor is -21).
 */
export function e4OperandTable(): Uint32Array {
  const table = new Uint32Array(256);
  for (let code = 0; code < 256; ++code) {
    const magnitude = code & 0x7f;
    const zero = magnitude === 0 || magnitude === 0x7f;
    const bits = f32Bits(e4m3ToNumber(code));
    if ((bits & 0xfffff) !== 0) throw new Error(`E4M3 code ${code} has more than 4 significant bits`);
    const exponent = zero ? -100 : Math.max((magnitude >> 3) - 7, -6);
    table[code] = ((bits & 0xfff00000) | (exponent + E4_OPERAND_EXPONENT_BIAS)) >>> 0;
  }
  return table;
}

let operandAttribute: StorageBufferAttribute | null = null;

/** `e4OperandTable()` as a shared storage attribute (256 words). */
export function e4OperandTableAttribute(): StorageBufferAttribute {
  operandAttribute ??= attributeFromBytes(e4OperandTable());
  return operandAttribute;
}
