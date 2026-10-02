// Typed TSL literals and packed byte / half / word access over `storage(attribute, 'uint')`.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The packing conventions are the reference's
// (ports/browser-webgpu/src/passes.js tensors, src/matmul/packed-activation.js, src/window/packed-output.js):
// little-endian, value `i` of a byte tensor in bits `8 * (i % 4)` of word `i / 4`, half `i` in bits `16 * (i % 2)` of
// word `i / 2`.
//
// Why the literal helpers exist (design section 3):
//   R1  TSL casts both operands of `a.op(b)` to the type of `a`; a bare JS number is a `float`. `u()` / `i()` / `f()`
//       make every literal's type explicit.
//   R2  Comparing a `uint` node with a bare JS number compares in f32 - wrong above 2^24. Compare with `u(n)`.
//   R3  TSL prints a float literal with the shortest JS repr (exact only for f32-representable numbers), prints
//       `float(-0)` as `0.0`, a negative `uint` as `0u`, and rounds `int` literals. `f()` asserts f32-representable
//       and rejects -0; `u()` / `i()` assert integer range.

import { float, int, uint, uintBitsToFloat, unpackHalf2x16, select, floatBitsToUint } from 'three/tsl';

/** A TSL node. TSL is dynamically typed at build time; `any` stands in for its `Node` type. */
export type TSLNode = any;

/**
 * Branch-free choice: WGSL `select(ifFalse, ifTrue, cond)` (both operands evaluated, no control flow), stored in a
 * variable. Use this, not TSL's bare `select` (R14):
 *   * bare `select` emits an `if/else` that re-generates each operand's whole subtree inside both branches, so nested
 *     selects grow exponentially (an 8-term FDPA became 3500 lines);
 *   * `select(...).uniformFlow()` emits the WGSL builtin but, when the node is used twice, the second use reads a
 *     temporary that was never assigned (three 0.186 `ConditionalNode.generate`). `.toVar()` makes every use read
 *     one assigned variable.
 * Argument order is TSL's: condition first (the reverse of WGSL's).
 */
export const pick = (cond: TSLNode, ifTrue: TSLNode, ifFalse: TSLNode): TSLNode =>
  select(cond, ifTrue, ifFalse).uniformFlow().toVar();

/** A `u32` literal. Throws unless `value` is an integer in [0, 2^32). */
export function u(value: number): TSLNode {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError(`u(): ${value} is not a u32`);
  }
  return uint(value);
}

/** An `i32` literal. Throws unless `value` is an integer in [-2^31, 2^31). */
export function i(value: number): TSLNode {
  if (!Number.isInteger(value) || value < -0x80000000 || value > 0x7fffffff) {
    throw new RangeError(`i(): ${value} is not an i32`);
  }
  return int(value);
}

/**
 * An `f32` literal. Throws unless `value` is finite, exactly representable in f32, and not -0 (TSL would print `0.0`;
 * use `fBits(0x80000000)`). Precompute constant subexpressions in JS with `Math.fround` (R8).
 */
export function f(value: number): TSLNode {
  if (!Number.isFinite(value) || Math.fround(value) !== value) {
    throw new RangeError(`f(): ${value} is not exactly representable as a finite f32`);
  }
  if (Object.is(value, -0)) throw new RangeError('f(): -0 would print as 0.0; use fBits(0x80000000)');
  return float(value);
}

/** An `f32` from its bit pattern (NaN payloads, -0, magic constants). */
export const fBits = (bits: number): TSLNode => uintBitsToFloat(u(bits >>> 0));

/** `u32` bit pattern of an f32 node. */
export const bitsOf = (value: TSLNode): TSLNode => floatBitsToUint(value);

/** Word `index` of a `uint` storage node. */
export const loadWord = (buffer: TSLNode, index: TSLNode): TSLNode => buffer.element(index);

/** Byte `byteIndex` (`uint` node) of a packed byte tensor (E4M3 codes), as a `uint` in [0, 255]. */
export function loadE4(buffer: TSLNode, byteIndex: TSLNode): TSLNode {
  const word = buffer.element(byteIndex.shiftRight(u(2)));
  return word.shiftRight(byteIndex.bitAnd(u(3)).shiftLeft(u(3))).bitAnd(u(0xff));
}

/** Half `halfIndex` (`uint` node) of a packed half tensor, as its bit pattern (`uint` in [0, 65535]). */
export function loadHalfBits(buffer: TSLNode, halfIndex: TSLNode): TSLNode {
  const word = buffer.element(halfIndex.shiftRight(u(1)));
  return word.shiftRight(halfIndex.bitAnd(u(1)).shiftLeft(u(4))).bitAnd(u(0xffff));
}

/**
 * Half `halfIndex` of a packed half tensor as an f32 value, through WGSL `unpack2x16float` (exact for every half,
 * subnormals included; verified exhaustively by numerics.gpu.test.ts).
 */
export function loadHalf(buffer: TSLNode, halfIndex: TSLNode): TSLNode {
  const pair = unpackHalf2x16(buffer.element(halfIndex.shiftRight(u(1))));
  return pick(halfIndex.bitAnd(u(1)).equal(u(0)), pair.x, pair.y);
}

/** The f32 stored at word `index` of an f32 tensor viewed as `uint`. */
export const loadF32 = (buffer: TSLNode, index: TSLNode): TSLNode => uintBitsToFloat(buffer.element(index));

/** Byte `lane` (JS number 0..3) of a word. */
export const byteOf = (word: TSLNode, lane: number): TSLNode => word.shiftRight(u(lane * 8)).bitAnd(u(0xff));

/** One word from four bytes (`uint` nodes in [0, 255]), `b0` in the low byte. */
export const packWord4 = (b0: TSLNode, b1: TSLNode, b2: TSLNode, b3: TSLNode): TSLNode =>
  b0
    .bitOr(b1.shiftLeft(u(8)))
    .bitOr(b2.shiftLeft(u(16)))
    .bitOr(b3.shiftLeft(u(24)));

/** One word from two half bit patterns (`uint` nodes in [0, 65535]), `lo` in the low half. */
export const packHalfPair = (lo: TSLNode, hi: TSLNode): TSLNode => lo.bitOr(hi.shiftLeft(u(16)));
