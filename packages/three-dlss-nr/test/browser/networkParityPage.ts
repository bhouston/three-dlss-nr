// Browser page of the full-network parity gate: the reference WebGPU port of OpenDLSS-NR (by maan, MIT;
// reference/OpenDLSS-NR/ports/browser-webgpu, its own `Network`, WGSL fetched from the submodule) and this port's
// `NRNetwork` (TSL) on ONE device in one page, same synthetic weights, same input features.
//
// Part of three-dlss-nr. Bundled and driven by run-network-parity-chrome.mjs (real Chrome: hardware f16 on D3D12 + DXC,
// which the reference needs and which Dawn in Node lacks on the dev machine). Exposes `globalThis.nrNetworkParity =
// { parity, bisect }`:
//   parity({ modelUrl, shaderBase, width, height }): every captured boundary, the three post tensors and the f32 head,
//     byte for byte; both networks run twice (repeat runs must be identical); our digest (the golden table for the
//     Node test); E4 statistics of the reference's boundaries (the synthetic-weights calibration gate).
//   bisect({ ..., until }): both networks truncated after dispatch `until`; every tensor label both allocate, compared.

import { Model } from '@ref/model.js';
import { Network } from '@ref/network.js';
import { WebGPURenderer } from 'three/webgpu';

import { createNRDevice } from '../../src/device.js';
import { NRModel } from '../../src/model/Model.js';
import { NRNetwork } from '../../src/NRNetwork.js';
import { e4m3ToNumber } from '../../src/numerics/oracle.js';
import { syntheticFeatures } from '../../src/synthetic/features.js';
import { digestTensors, POST_TENSORS, readParityTensors } from '../network/golden.js';

const log = (...args: unknown[]) => console.log(...args);

interface Options {
  modelUrl: string;
  shaderBase: string;
  width: number;
  height: number;
}

export interface Verdict {
  name: string;
  bytes: number;
  mismatches: number;
  /** First differing byte (-1 when equal); for E4 tensors also the codes there. */
  first: number;
  ours?: number;
  reference?: number;
}

export interface BoundaryStats {
  name: string;
  saturated: number;
  zeros: number;
  median: number;
  distinct: number;
}

function verdict(name: string, ours: Uint8Array, reference: Uint8Array): Verdict {
  let mismatches = Math.abs(ours.length - reference.length);
  let first = -1;
  const length = Math.min(ours.length, reference.length);
  for (let i = 0; i < length; ++i) {
    if (ours[i] !== reference[i]) {
      if (first < 0) first = i;
      mismatches += 1;
    }
  }
  return first < 0
    ? { name, bytes: ours.length, mismatches, first }
    : { name, bytes: ours.length, mismatches, first, ours: ours[first], reference: reference[first] };
}

/** Statistics of one E4 boundary (`syntheticStats.gpu.test.ts`'s gate). */
function e4Stats(name: string, bytes: Uint8Array): BoundaryStats {
  let saturated = 0;
  let zeros = 0;
  const distinct = new Set<number>();
  const magnitudes = new Float64Array(bytes.length);
  bytes.forEach((code, i) => {
    if ((code & 0x7f) === 0x7e) saturated += 1;
    if ((code & 0x7f) === 0) zeros += 1;
    distinct.add(code);
    magnitudes[i] = Math.abs(e4m3ToNumber(code));
  });
  magnitudes.sort();
  return {
    name,
    saturated: saturated / bytes.length,
    zeros: zeros / bytes.length,
    median: magnitudes[magnitudes.length >> 1],
    distinct: distinct.size,
  };
}

async function setup({ modelUrl, shaderBase, width, height }: Options, after?: (network: any) => void) {
  const { device } = await createNRDevice();
  if (!device.features.has('shader-f16'))
    throw new Error('this browser/device has no shader-f16; the reference needs it');
  const renderer = new WebGPURenderer({ device, antialias: false });
  await renderer.init();
  const info = (device as any).adapterInfo ?? {};
  const adapter = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim();
  log(`device: ${adapter}`);

  log('loading weights (reference Model and NRModel)');
  const referenceModel = await new Model(device).load(modelUrl.replace(/\/+$/, ''));
  const model = await NRModel.load(modelUrl);
  log('creating the reference network');
  const reference = await Network.create({
    device,
    model: referenceModel,
    width,
    height,
    captureBoundaries: true,
    shaderBase: new URL(shaderBase, location.href),
    after,
  });
  log('creating the TSL network (compiling)');
  let lastLogged = 0;
  const ours = await NRNetwork.create({
    renderer,
    model,
    width,
    height,
    captureBoundaries: true,
    onProgress: (message, progress) => {
      if (progress.phase !== 'compiling' || progress.loaded! - lastLogged >= 50 || progress.loaded === progress.total) {
        lastLogged = progress.loaded ?? 0;
        log(message);
      }
    },
  });
  const features = syntheticFeatures(ours.geometry);
  reference.writeFeatures(features);
  ours.writeFeatures(features);
  const referenceReader = {
    boundaryNames: reference.boundaryNames as string[],
    readBoundary: (name: string) => reference.readBoundary(name) as Promise<Uint8Array>,
    readTensor: async (label: string) => (await reference.readTensorByLabel(label)).bytes as Uint8Array,
    readHead: () => reference.readHead() as Promise<Float32Array>,
  };
  const dispose = () => {
    ours.dispose();
    reference.destroy();
    referenceModel.destroy();
    model.dispose();
    renderer.dispose();
    device.destroy();
  };
  return { adapter, reference, referenceReader, ours, dispose };
}

async function parity(options: Options) {
  const { adapter, reference, referenceReader, ours, dispose } = await setup(options);
  try {
    log('running both networks');
    await reference.run();
    await ours.run();
    const theirs = await readParityTensors(referenceReader);
    const mine = await readParityTensors(ours);
    if ([...theirs.keys()].join() !== [...mine.keys()].join()) {
      throw new Error(`tensor names differ:\n${[...theirs.keys()].join()}\n${[...mine.keys()].join()}`);
    }
    const results = [...mine].map(([name, bytes]) => verdict(name, bytes, theirs.get(name)!));
    const boundaryStats = [...theirs]
      .filter(([name]) => name !== 'head' && !(POST_TENSORS as readonly string[]).includes(name))
      .map(([name, bytes]) => e4Stats(name, bytes));
    const head = new Float32Array(theirs.get('head')!.buffer);
    const finiteHead = head.every(Number.isFinite);

    log('running both networks again (determinism)');
    await reference.run();
    await ours.run();
    const theirsAgain = await readParityTensors(referenceReader);
    const mineAgain = await readParityTensors(ours);
    const repeat = {
      reference: [...theirs].filter(([name, bytes]) => verdict(name, theirsAgain.get(name)!, bytes).mismatches).length,
      ours: [...mine].filter(([name, bytes]) => verdict(name, mineAgain.get(name)!, bytes).mismatches).length,
    };
    const digest = await digestTensors(mine);
    return {
      adapter,
      width: options.width,
      height: options.height,
      field: `${ours.geometry.fullWidth}x${ours.geometry.fullHeight}`,
      dispatches: ours.dispatchLabels.length,
      results,
      repeat,
      finiteHead,
      stats: boundaryStats,
      digest,
    };
  } finally {
    dispose();
  }
}

/** Truncate the reference's recorder after its `cut`-th dispatch, keeping the captures that follow it. */
function truncateReference(cut: number) {
  return (network: any) => {
    const passes = network.recorder.passes as { kind: string }[];
    let dispatches = 0;
    let end = passes.length;
    for (const [i, pass] of passes.entries()) {
      if (pass.kind !== 'dispatch') continue;
      if (dispatches === cut) {
        end = i;
        break;
      }
      dispatches += 1;
    }
    passes.length = end;
  };
}

/** The reference alone (no TSL compile): its boundary statistics, for calibrating the synthetic weights quickly. */
async function stats({ modelUrl, shaderBase, width, height }: Options) {
  const { device } = await createNRDevice();
  try {
    const referenceModel = await new Model(device).load(modelUrl.replace(/\/+$/, ''));
    const reference = await Network.create({
      device,
      model: referenceModel,
      width,
      height,
      captureBoundaries: true,
      shaderBase: new URL(shaderBase, location.href),
    });
    reference.writeFeatures(syntheticFeatures(reference.geometry));
    await reference.run();
    const result: BoundaryStats[] = [];
    for (const name of reference.boundaryNames as string[])
      result.push(e4Stats(name, await reference.readBoundary(name)));
    const head = (await reference.readHead()) as Float32Array;
    reference.destroy();
    referenceModel.destroy();
    return { stats: result, finiteHead: head.every(Number.isFinite) };
  } finally {
    device.destroy();
  }
}

async function bisect(options: Options & { until: number }) {
  const { adapter, reference, ours, dispose } = await setup(options, truncateReference(options.until));
  try {
    await reference.run();
    await ours.run({ until: options.until });
    const results: Verdict[] = [];
    const labels = [...new Set([...ours.graph.tensors.byKey.values()].map((t) => t.label))];
    for (const label of labels) {
      let theirs: Uint8Array;
      try {
        theirs = (await reference.readTensorByLabel(label)).bytes;
      } catch {
        continue;
      }
      results.push(verdict(label, await ours.readTensor(label), theirs));
    }
    return { adapter, until: options.until, label: ours.dispatchLabels[options.until - 1], results };
  } finally {
    dispose();
  }
}

/** Reject as soon as the device is lost (e.g. a TDR on a busy GPU) instead of hanging the harness. */
function failOnDeviceLoss<A extends unknown[], R>(run: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
  return (...args: A) =>
    new Promise<R>((resolve, reject) => {
      const original = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
        const device = await original.call(this, descriptor);
        void device.lost.then((info) => {
          if (info.reason !== 'destroyed') reject(new Error(`WebGPU device lost: ${info.message}`));
        });
        return device;
      };
      run(...args).then(resolve, reject);
    });
}

(globalThis as any).nrNetworkParity = {
  parity: failOnDeviceLoss(parity),
  bisect: failOnDeviceLoss(bisect),
  stats: failOnDeviceLoss(stats),
};
