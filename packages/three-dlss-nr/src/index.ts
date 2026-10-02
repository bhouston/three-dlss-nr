// three-dlss-nr: a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan
// (https://github.com/maanHimself/OpenDLSS-NR, MIT, pinned at 9d08f41).
// Not affiliated with NVIDIA. "DLSS" is an NVIDIA trademark used descriptively;
// no NVIDIA weights are included. See LICENSE and NOTICE.

/** The upstream implementation this package ports. */
export const UPSTREAM = {
  name: 'OpenDLSS-NR',
  author: 'maan',
  url: 'https://github.com/maanHimself/OpenDLSS-NR',
  commit: '9d08f41',
  license: 'MIT',
} as const;

/** Package version (placeholder until the network port lands). */
export const VERSION = '0.1.0';

export type {
  ComputeNode,
  Dim3,
  F16Matrix,
  F32Vector,
  FP8Matrix,
  GemmF16Buffers,
  GemmF16Spec,
  GemmFp8Buffers,
  GemmSpec,
  HalfVector,
  NRKernel,
  NRTensor,
  StorageBufferAttribute,
  TensorFormat,
  VitAttendBuffers,
  VitNormalizeBuffers,
  VitSpec,
  WindowAttentionBuffers,
  WindowAttentionSpec,
} from './types.js';

export {
  checkNRDeviceLimits,
  createNRDevice,
  createNRRenderer,
  NR_MIN_WORKGROUP_STORAGE,
  nrDeviceProblems,
  type NRDevice,
  type NRDeviceOptions,
  type NRRendererOptions,
} from './device.js';

export {
  alignUp,
  fusedLayout,
  geometryFromValid,
  grid1d,
  postFusedLayout,
  preFusedLayout,
  upsampleFusedLayout,
  windowPhase,
  WindowPhases,
  type FusedLayout,
  type NRGeometry,
  type NRLevel,
} from './geometry.js';

export {
  attributeBytes,
  attributeFromBytes,
  bytesPerValue,
  createF32Vector,
  createHalfVector,
  createTensor,
  fillBuffer,
  NRTensors,
  readBuffer,
  wordAttribute,
  writeBuffer,
} from './tensors.js';

// Backend-neutral network interface (the TSL port and the reference shim, `three-dlss-nr/reference-backend`).
export {
  rendererDevice,
  unmetRequirements,
  type NRBackend,
  type NRBackendCreateOptions,
  type NRBackendFactory,
  type NRBackendGeometry,
  type NRBackendId,
  type NRBackendLimit,
  type NRBackendMemory,
  type NRBackendModelSource,
  type NRBackendRequirements,
  type NRFrameTiming,
  type NRManifestLike,
  type NRModelFilesLike,
  type NRModelStagesLike,
  type NRRunOptions,
  type NRTimingMethod,
} from './backend/NRBackend.js';
export { NRFrameTimer, summarizeMilliseconds } from './backend/timing.js';
