// KernelBuilder and the typed literals, without a GPU: the R1-R3 and R7 guards.

import { describe, expect, it } from 'vitest';

import { createTensor, wordAttribute } from '../tensors.js';
import { kernel } from './KernelBuilder.js';
import { f, fBits, i, u } from './packed.js';

describe('typed literals (R1-R3)', () => {
  it('accept exactly representable values', () => {
    expect(() => u(0)).not.toThrow();
    expect(() => u(0xffffffff)).not.toThrow();
    expect(() => i(-0x80000000)).not.toThrow();
    expect(() => i(0x7fffffff)).not.toThrow();
    expect(() => f(0.044921875)).not.toThrow();
    expect(() => f(Math.fround(0.1))).not.toThrow();
    expect(() => fBits(0x80000000)).not.toThrow();
  });

  it('reject values that would print as a different literal', () => {
    expect(() => u(-1)).toThrow(RangeError); // TSL prints a negative uint as 0u
    expect(() => u(2 ** 32)).toThrow(RangeError);
    expect(() => u(1.5)).toThrow(RangeError);
    expect(() => i(2 ** 31)).toThrow(RangeError); // invalid i32 literal
    expect(() => i(0.5)).toThrow(RangeError); // TSL would Math.round it
    expect(() => f(0.1)).toThrow(RangeError); // not an f32: Tint would round the literal
    expect(() => f(-0)).toThrow(RangeError); // TSL prints float(-0) as 0.0
    expect(() => f(Number.NaN)).toThrow(RangeError);
    expect(() => f(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});

describe('kernel()', () => {
  const a = createTensor('a', 64, 32, 'e4');
  const b = createTensor('b', 64, 32, 'f16');
  const base = { label: 'test', kind: 'test', workgroupSize: [64], dispatch: [1], body: () => {} };

  it('records reads and writes and pads sizes to three components', () => {
    const k = kernel({ ...base, inputs: { a }, outputs: { b } });
    expect(k.reads).toEqual([a]);
    expect(k.writes).toEqual([b]);
    expect(k.dispatch).toEqual([1, 1, 1]);
    expect(k.workgroupSize).toEqual([64, 1, 1]);
    expect(k.node.isComputeNode).toBe(true);
  });

  it('refuses one buffer bound twice, or as both input and output (R7)', () => {
    expect(() => kernel({ ...base, inputs: { a }, outputs: { out: a } })).toThrow(/same buffer/);
    expect(() => kernel({ ...base, inputs: { a, again: a }, outputs: { b } })).toThrow(/same buffer/);
  });

  it('refuses dispatches beyond 65535 groups, empty outputs and non-u32 buffers', () => {
    expect(() => kernel({ ...base, dispatch: [65536], inputs: {}, outputs: { b } })).toThrow(/65535/);
    expect(() => kernel({ ...base, dispatch: [0], inputs: {}, outputs: { b } })).toThrow(RangeError);
    expect(() => kernel({ ...base, inputs: { a }, outputs: {} })).toThrow(/without outputs/);
    expect(() => kernel({ ...base, inputs: {}, outputs: { x: { attribute: {} as never } } })).toThrow(TypeError);
    expect(() => kernel({ ...base, inputs: {}, outputs: { x: { attribute: wordAttribute(16) } } })).not.toThrow();
  });
});
