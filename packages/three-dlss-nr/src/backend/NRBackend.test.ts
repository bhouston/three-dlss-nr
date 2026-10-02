import { describe, expect, it } from 'vitest';

import { backendProblems, factoryProblems } from '../../test/backendConformance.js';
import {
  REFERENCE_REQUIREMENTS,
  ReferenceWgslBackend,
  referenceUnavailableReason,
  referenceWgslBackend,
} from '../reference-backend/index.js';
import { createTensor } from '../tensors.js';
import type { NRBackend, NRBackendFactory } from './NRBackend.js';
import { rendererDevice, unmetRequirements } from './NRBackend.js';
import { summarizeMilliseconds } from './timing.js';

/** A device-shaped object with the given features and limits. */
const fakeDevice = (features: string[], limits: Record<string, number>) =>
  ({ features: new Set(features), limits }) as unknown as GPUDevice;

const capable = {
  maxComputeWorkgroupStorageSize: 32768,
  maxStorageBuffersPerShaderStage: 10,
  maxComputeInvocationsPerWorkgroup: 1024,
};

// Static: the class is assignable to the interface (a type error otherwise).
const asBackend = (backend: ReferenceWgslBackend): NRBackend => backend;

describe('NRBackend interface', () => {
  it('the reference factory conforms', () => {
    const factory: NRBackendFactory<ReferenceWgslBackend> = referenceWgslBackend;
    expect(factoryProblems(factory)).toEqual([]);
    expect(factory.id).toBe('reference-wgsl');
  });

  it('ReferenceWgslBackend implements every NRBackend member', () => {
    void asBackend;
    // Runtime: a shell with the fields `create` sets passes the conformance check.
    const features = createTensor('input features', 336 * 320, 16, 'f32');
    const head = createTensor('head', 336 * 320, 4, 'f32');
    const shell = Object.assign(Object.create(ReferenceWgslBackend.prototype), {
      id: 'reference-wgsl',
      label: 'x',
      requirements: REFERENCE_REQUIREMENTS,
      geometry: { validWidth: 64, validHeight: 64, fullWidth: 336, fullHeight: 320, fullRows: 336 * 320 },
      features,
      head,
      network: { recorder: { dispatchCount: 451, passes: [] }, boundaryNames: [], tensors: { total: 1 } },
      referenceModel: { bytesUploaded: 1 },
    });
    expect(backendProblems(shell)).toEqual([]);
  });

  it('reports unmet requirements by name', () => {
    expect(unmetRequirements(fakeDevice(['shader-f16'], capable), REFERENCE_REQUIREMENTS)).toEqual([]);
    const problems = unmetRequirements(
      fakeDevice([], { ...capable, maxComputeWorkgroupStorageSize: 16384 }),
      REFERENCE_REQUIREMENTS,
    );
    expect(problems).toEqual([
      "the device lacks the 'shader-f16' feature",
      'maxComputeWorkgroupStorageSize is 16384; needs at least 32768',
    ]);
  });

  it('explains how to get shader-f16 when it is missing', () => {
    expect(referenceUnavailableReason(fakeDevice(['shader-f16'], capable))).toBeNull();
    const reason = referenceUnavailableReason(fakeDevice([], capable))!;
    expect(reason).toContain("'shader-f16'");
    expect(reason).toContain('createNRDevice');
    expect(reason).toContain('requiredLimits');
    expect(referenceWgslBackend.unavailableReason({})).toBe('the renderer has no WebGPU device');
  });

  it('rendererDevice rejects a renderer without a WebGPU device', () => {
    expect(() => rendererDevice({ backend: {} })).toThrow(/no WebGPU device/);
    const device = fakeDevice([], {});
    expect(rendererDevice({ backend: { device } })).toBe(device);
  });

  it('summarizes timings', () => {
    expect(summarizeMilliseconds([])).toBeNull();
    expect(summarizeMilliseconds([null, 3, 1, 2])).toEqual({ min: 1, median: 2 });
    expect(summarizeMilliseconds([4, 1, 3, 2])).toEqual({ min: 1, median: 2.5 });
  });
});
