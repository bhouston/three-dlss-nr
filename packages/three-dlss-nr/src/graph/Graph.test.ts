// The graph's pass list against a record-only run of the reference graph (OpenDLSS-NR's WebGPU port, by maan, MIT;
// ports/browser-webgpu/src/graph.js), on the synthetic model: same dispatches in the same order with the same labels,
// kinds and workgroup counts, the same boundary captures after the same dispatches, and the same tensors.

import { beforeAll, describe, expect, it } from 'vitest';

import { registerSyntheticFiles } from '../../test/setup/fetchShim.js';
import { fakeDevice, loadReferenceModelOn, recordGraphRequests } from '../../test/reference/refModel.js';
import { geometryFromValid } from '../geometry.js';
import { NRModel } from '../model/Model.js';
import { generateSyntheticModel } from '../synthetic/generate.js';
import { NRGraph } from './Graph.js';

let ours: NRModel;
let reference: any;

beforeAll(async () => {
  const synthetic = await generateSyntheticModel({ seed: 1 });
  ours = await NRModel.load(synthetic);
  reference = await loadReferenceModelOn(fakeDevice().device, registerSyntheticFiles('nr-graph', synthetic.files));
}, 60_000);

describe('NRGraph', () => {
  it.for([
    [64, 64],
    [300, 500],
    [512, 512],
  ] as const)('records the reference pass list at %ix%i', ([width, height]) => {
    const expected = recordGraphRequests(reference, width, height, { captureBoundaries: true });
    const graph = new NRGraph(ours, geometryFromValid(width, height), { captureBoundaries: true });
    expect(graph.dispatches).toHaveLength(451);
    expect(expected.passes).toHaveLength(451);
    const actual = graph.dispatches.map((k) => ({ label: k.label, kind: k.kind, dispatch: [...k.dispatch] }));
    expect(actual).toEqual(expected.passes);
    const captures = graph.passes.filter((p) => p.capture).map((p) => `${p.index} capture ${p.capture}`);
    expect(captures).toEqual(expected.captures);
    // block-0 .. block-69 (70; block-39 is the merge) + 5 transition-a-b + 4 pooled-a-b = 79.
    expect(captures).toHaveLength(79);
    // Every capture runs right after the dispatch it captures.
    graph.passes.forEach((pass, i) => {
      if (pass.capture) expect(graph.passes[i - 1].index).toBe(pass.index);
    });
  });

  it('allocates the reference tensors (label, shape, format) and nothing else', () => {
    const graph = new NRGraph(ours, geometryFromValid(64, 64), { captureBoundaries: true });
    expect([...graph.tensors.byKey.keys()].toSorted()).toEqual(referenceTensorKeys(64, 64));
  });

  it('is identical across rebuilds (labels, sizes) and without captures has no copies', () => {
    const geometry = geometryFromValid(64, 64);
    const a = new NRGraph(ours, geometry);
    const b = new NRGraph(ours, geometry);
    expect(a.passes).toHaveLength(451);
    expect(a.labels).toEqual(b.labels);
    expect(a.boundaries.size).toBe(0);
    expect(a.head.label).toBe('head');
    expect(a.features.channels).toBe(16);
  });
});

/** The sorted tensor keys the reference allocates for a valid size: its graph's, plus `input features` (network.js). */
function referenceTensorKeys(width: number, height: number): string[] {
  const keys = new Set<string>([`input features/${geometryFromValid(width, height).fullRows}x16/f32`]);
  recordGraphRequests(reference, width, height, {
    captureBoundaries: true,
    onAllocate: (label, rows, channels, format) => keys.add(`${label}/${rows}x${channels}/${format}`),
  });
  return [...keys].toSorted();
}
