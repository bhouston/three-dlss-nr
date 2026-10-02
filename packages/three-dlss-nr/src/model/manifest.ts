// The model directory's manifest: types, structural validation, and the graph's byte-length checks.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The format is the reference's (docs/weights.md,
// read by ports/browser-webgpu/src/model.js `Model.load`): `manifest.json` with `totals.blockCount`, eleven stage
// files of packed bytes, and 153 `blockN.layerM.parameter` records, each a `(stage, stageOffset, byteLength)` slice.
// The record checks are the ones the reference graph makes when it records (graph.js:365, 518, 558) plus "every
// region the graph reads lies inside its record" for the others.

import { modelRecords, NR_BLOCK_COUNT } from './layouts.js';

export interface NRManifestStage {
  readonly id: string;
  /** Path relative to `<directory>/model/` (the reference fetches `${directory}/model/${file}`). */
  readonly file: string;
  readonly packedByteLength: number;
  /** Lowercase hex SHA-256 of the stage file. */
  readonly sha256?: string;
}

export interface NRManifestTensor {
  readonly name: string;
  readonly block: number;
  readonly layer: number;
  readonly stage: string;
  readonly stageOffset: number;
  readonly byteLength: number;
}

export interface NRManifest {
  /** Free text; the synthetic generator writes `three-dlss-nr synthetic v1`. */
  readonly format?: string;
  /** The synthetic generator's seed. */
  readonly seed?: number;
  readonly totals: { readonly blockCount: number };
  readonly stages: readonly NRManifestStage[];
  readonly tensors: readonly NRManifestTensor[];
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isCount = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;

/** Check the shape of a parsed `manifest.json` and return it typed. Throws on the first malformed field. */
export function parseManifest(value: unknown): NRManifest {
  if (!isObject(value)) throw new Error('manifest: not an object');
  const { totals, stages, tensors } = value;
  if (!isObject(totals) || !isCount(totals.blockCount)) throw new Error('manifest: totals.blockCount missing');
  if (!Array.isArray(stages)) throw new Error('manifest: stages is not an array');
  if (!Array.isArray(tensors)) throw new Error('manifest: tensors is not an array');
  stages.forEach((stage: unknown, index) => {
    if (
      !isObject(stage) ||
      typeof stage.id !== 'string' ||
      typeof stage.file !== 'string' ||
      !isCount(stage.packedByteLength) ||
      (stage.sha256 !== undefined && (typeof stage.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(stage.sha256)))
    ) {
      throw new Error(`manifest: stage ${index} is malformed`);
    }
  });
  tensors.forEach((tensor: unknown, index) => {
    if (
      !isObject(tensor) ||
      typeof tensor.name !== 'string' ||
      !isCount(tensor.block) ||
      !isCount(tensor.layer) ||
      typeof tensor.stage !== 'string' ||
      !isCount(tensor.stageOffset) ||
      !isCount(tensor.byteLength)
    ) {
      throw new Error(`manifest: tensor ${index} is malformed`);
    }
  });
  return value as unknown as NRManifest;
}

/**
 * Everything wrong with a manifest as a model of the 71-block network, as messages (empty when it is usable):
 * block count, duplicate or unknown stages, records past their stage, and for each of the 153 records the graph
 * reads, its presence and length. Extra records are allowed and ignored.
 */
export function manifestProblems(manifest: NRManifest): string[] {
  const problems: string[] = [];
  if (manifest.totals.blockCount !== NR_BLOCK_COUNT) {
    problems.push(`the model has ${manifest.totals.blockCount} blocks; this network has ${NR_BLOCK_COUNT}`);
  }
  const stages = new Map<string, NRManifestStage>();
  for (const stage of manifest.stages) {
    if (stages.has(stage.id)) problems.push(`duplicate stage ${stage.id}`);
    stages.set(stage.id, stage);
  }
  const tensors = new Map<string, NRManifestTensor>();
  for (const tensor of manifest.tensors) {
    if (tensors.has(tensor.name)) problems.push(`duplicate tensor ${tensor.name}`);
    tensors.set(tensor.name, tensor);
    const stage = stages.get(tensor.stage);
    if (!stage) problems.push(`tensor ${tensor.name} references unknown stage ${tensor.stage}`);
    else if (tensor.stageOffset + tensor.byteLength > stage.packedByteLength) {
      problems.push(`tensor ${tensor.name} exceeds stage ${tensor.stage}`);
    }
  }
  for (const expected of modelRecords()) {
    const tensor = tensors.get(expected.name);
    if (!tensor) {
      problems.push(`missing tensor ${expected.name}`);
    } else if (expected.exact ? tensor.byteLength !== expected.byteLength : tensor.byteLength < expected.readEnd) {
      problems.push(
        `tensor ${expected.name} is ${tensor.byteLength} bytes; the graph needs ` +
          `${expected.exact ? 'exactly' : 'at least'} ${expected.exact ? expected.byteLength : expected.readEnd}`,
      );
    } else if (tensor.block !== expected.block || tensor.layer !== expected.layer) {
      problems.push(`tensor ${expected.name} is labelled block ${tensor.block} layer ${tensor.layer}`);
    }
  }
  return problems;
}

/** Throw unless `manifestProblems` is empty. */
export function validateManifest(manifest: NRManifest): void {
  const problems = manifestProblems(manifest);
  if (problems.length) {
    const shown = problems.slice(0, 8).join('; ');
    throw new Error(`invalid model manifest: ${shown}${problems.length > 8 ? ` (+${problems.length - 8} more)` : ''}`);
  }
}

/** Lowercase hex SHA-256 (WebCrypto, available in browsers and Node >= 19). */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}
