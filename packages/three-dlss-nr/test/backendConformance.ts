// Conformance checks for `NRBackend` implementations (the TSL port and the reference-wgsl shim).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). `NRBACKEND_MEMBERS` lists every member of
// the interface; `backendProblems(instance)` checks a live backend against it and its invariants, so chunk E's
// `NRNetwork` can reuse the same test.

import type { NRBackend, NRBackendFactory } from '../src/backend/NRBackend.js';

/** Methods of `NRBackend`. */
export const NRBACKEND_METHODS = [
  'writeFeatures',
  'run',
  'readHead',
  'readBoundary',
  'readTensor',
  'dispose',
] as const satisfies readonly (keyof NRBackend)[];

/** Properties of `NRBackend`. */
export const NRBACKEND_PROPERTIES = [
  'id',
  'label',
  'requirements',
  'geometry',
  'dispatchCount',
  'features',
  'head',
  'boundaryNames',
  'memory',
] as const satisfies readonly (keyof NRBackend)[];

// Compile-time completeness: adding a member to NRBackend without listing it here fails `pnpm tsc`.
type Listed = (typeof NRBACKEND_METHODS)[number] | (typeof NRBACKEND_PROPERTIES)[number];
const complete: Record<Exclude<keyof NRBackend, Listed>, never> = {};
void complete;

/** Methods of `NRBackendFactory`. */
export const NRBACKEND_FACTORY_MEMBERS = [
  'id',
  'label',
  'requirements',
  'unavailableReason',
  'create',
] as const satisfies readonly (keyof NRBackendFactory)[];

/** Problems with a factory object (empty when it conforms). */
export function factoryProblems(factory: NRBackendFactory): string[] {
  const problems: string[] = [];
  for (const name of NRBACKEND_FACTORY_MEMBERS) if (!(name in factory)) problems.push(`factory lacks ${name}`);
  if (factory.id !== 'tsl' && factory.id !== 'reference-wgsl') problems.push(`unknown id ${factory.id}`);
  if (!Array.isArray(factory.requirements?.features)) problems.push('requirements.features is not an array');
  return problems;
}

/** Problems with a created backend (empty when it conforms). */
export function backendProblems(backend: NRBackend): string[] {
  const problems: string[] = [];
  for (const name of NRBACKEND_METHODS) {
    if (typeof backend[name] !== 'function') problems.push(`${name} is not a method`);
  }
  for (const name of NRBACKEND_PROPERTIES) if (backend[name] === undefined) problems.push(`${name} is undefined`);
  const g = backend.geometry;
  if (g.fullRows !== g.fullWidth * g.fullHeight) problems.push('geometry.fullRows != fullWidth * fullHeight');
  if (g.fullWidth < g.validWidth || g.fullHeight < g.validHeight) problems.push('field smaller than the valid size');
  const tensor = (name: 'features' | 'head', channels: number) => {
    const t = backend[name];
    if (t.rows !== g.fullRows || t.channels !== channels || t.format !== 'f32') {
      problems.push(`${name} is ${t.rows}x${t.channels}/${t.format}; expected ${g.fullRows}x${channels}/f32`);
    }
    if (!t.attribute?.isStorageBufferAttribute) problems.push(`${name} has no storage attribute`);
  };
  tensor('features', 16);
  tensor('head', 4);
  if (!(backend.dispatchCount > 0)) problems.push('dispatchCount is not positive');
  if (!(backend.memory.activationBytes > 0) || !(backend.memory.weightBytes > 0)) problems.push('memory not reported');
  return problems;
}
