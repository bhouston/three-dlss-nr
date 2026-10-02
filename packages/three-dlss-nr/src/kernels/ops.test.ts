// The elementwise, preprocess and frame factories without a GPU: dispatch sizes as the reference graph records them
// (graph.js `op` uses grid1d(count / 4)), kinds and labels, reads / writes metadata, and the argument checks.

import { describe, expect, it } from 'vitest';

import { NRFrameParams } from '../frame/frameInputs.js';
import { createFrameKernels, NRHistory } from '../frame/history.js';
import { grid1d } from '../geometry.js';
import { createHalfVector, createTensor } from '../tensors.js';
import {
  createConvertF32ToF16,
  createCopyWords,
  createDownsample,
  createPostBlend,
  createUpsampleResidual,
} from './ops.js';
import { createPreprocess } from './preprocess.js';

/** Buffers of a small upsample_residual. */
const args = () => ({
  input: createTensor('low', 16, 32, 'f16'),
  skip: createTensor('skip', 64, 32, 'e4'),
  scale: createHalfVector(new Uint16Array(32)),
  output: createTensor('merged', 64, 32, 'e4'),
});

describe('ops factories', () => {
  it('dispatch one invocation per four values, as graph.js op records it (grid1d(count / 4))', () => {
    const input = createTensor('pool in', 1152 * 1024, 32, 'f16');
    const output = createTensor('pool out', 576 * 512, 32, 'e4');
    const k = createDownsample(
      { channels: 32, inWidth: 1152, inHeight: 1024, outWidth: 576, outHeight: 512, label: 'pool 0' },
      { input, output },
    );
    expect(k.kind).toBe('downsample');
    expect(k.label).toBe('pool 0');
    expect(k.dispatch).toEqual(grid1d((576 * 512 * 32) / 4));
    expect(k.workgroupSize).toEqual([64, 1, 1]);
    expect(k.reads).toEqual([input]);
    expect(k.writes).toEqual([output]);
  });

  it('record the dual output only when given', () => {
    const spec = { channels: 32, inWidth: 4, inHeight: 4, outWidth: 8, outHeight: 8, label: 'block 39 merge' };
    const single = createUpsampleResidual(spec, args());
    expect(single.writes.map((t) => t.label)).toEqual(['merged']);
    const raw = createTensor('merged raw', 64, 32, 'f16');
    const dual = createUpsampleResidual(spec, { ...args(), outputF16: raw });
    expect(dual.writes.map((t) => t.label)).toEqual(['merged', 'merged raw']);
    expect(dual.kind).toBe('upsample_residual');
  });

  it('reject wrong formats, short tensors and scale ranges', () => {
    const levels = { channels: 32, inWidth: 4, inHeight: 4, outWidth: 2, outHeight: 2, label: 'pool' };
    expect(() =>
      createDownsample(levels, { input: createTensor('in', 16, 32, 'e4'), output: createTensor('out', 4, 32, 'e4') }),
    ).toThrow(/expected f16/);
    expect(() =>
      createConvertF32ToF16(
        { count: 64 * 17, label: 'convert' },
        { input: createTensor('f', 64, 16, 'f32'), output: createTensor('h', 64, 16, 'f16') },
      ),
    ).toThrow(/exceeds/);
    expect(() =>
      createPostBlend(
        { ...levels, inWidth: 1, inHeight: 1, label: 'post blend' },
        {
          input: createTensor('in', 1, 32, 'e4'),
          skip: createTensor('skip', 4, 32, 'e4'),
          scales: createHalfVector(new Uint16Array(32)),
          output: createTensor('out', 4, 32, 'e4'),
        },
      ),
    ).toThrow(/scales exceed/);
    expect(() =>
      createDownsample(
        { ...levels, channels: 30 },
        { input: createTensor('in', 16, 30, 'f16'), output: createTensor('out', 4, 30, 'e4') },
      ),
    ).toThrow(/multiple of 4/);
  });

  it('copy words: every word of the source by default', () => {
    const source = createTensor('block-5', 100, 32, 'e4');
    const target = createTensor('boundary block-5', 100, 32, 'e4');
    const k = createCopyWords({ label: 'capture block-5' }, { source, target });
    expect(k.kind).toBe('copy_words');
    expect(k.dispatch).toEqual(grid1d(source.attribute.count));
    expect(() => createCopyWords({ words: 10 ** 6, label: 'too many' }, { source, target })).toThrow(/exceed/);
  });
});

describe('preprocess and frame factories', () => {
  const geometry = { fullWidth: 48, fullHeight: 40, validWidth: 40, validHeight: 30 };

  it('preprocess: 8x8 workgroups over the field', () => {
    const features = createTensor('input features', 48 * 40, 16, 'f32');
    const k = createPreprocess(
      { ...geometry, sourceWidth: 40, sourceHeight: 30, seed: 1 },
      { proxy: { attribute: createTensor('proxy', 40 * 30, 4, 'f32').attribute }, features },
    );
    expect(k.kind).toBe('preprocess');
    expect(k.dispatch).toEqual([6, 5, 1]);
    expect(k.workgroupSize).toEqual([8, 8, 1]);
    expect(k.writes).toEqual([features]);
  });

  it('frame kernels for both parities: labels, dispatches, and the history ping-pong', () => {
    const history = new NRHistory(40, 30);
    const features = createTensor('input features', 48 * 40, 16, 'f32');
    const head = createTensor('head', 48 * 40, 4, 'f32');
    const color = { buffer: { attribute: createTensor('scene', 40 * 30, 4, 'f16').attribute } };
    const frame = createFrameKernels(geometry, {
      color,
      motion: null,
      features,
      head,
      params: new NRFrameParams(),
      history,
    });
    expect(frame.inputFeatures.map((k) => k.label)).toEqual(['input features', 'input features']);
    expect(frame.compose.map((k) => k.kind)).toEqual(['compose', 'compose']);
    expect(frame.inputFeatures[0].dispatch).toEqual([6, 5, 1]);
    expect(frame.compose[0].dispatch).toEqual([5, 4, 1]);
    expect(frame.compose[0].reads).toEqual([head]);
    expect(history.write(0)).toBe(history.read(1));
    expect(history.write(1)).toBe(history.read(0));
    expect(() =>
      createFrameKernels(geometry, {
        color,
        motion: null,
        features,
        head,
        params: new NRFrameParams(),
        history: new NRHistory(8, 8),
      }),
    ).toThrow(/history is 8x8/);
  });

  it('frame params: settings map to uniform values', () => {
    const params = new NRFrameParams({ seed: 0x1_0005, historyValid: true, autoMask: true, blendScale: 0.7397 });
    expect(params.seed.value).toBe(0x1_0005);
    expect(params.historyValid.value).toBe(1);
    expect(params.autoMask.value).toBe(1);
    expect(params.blendScale.value).toBe(0.7397);
    params.set({ enabled: false });
    expect(params.enabled.value).toBe(0);
    expect(params.historyValid.value).toBe(1);
  });
});
