// three-dlss-nr/reference-backend: OpenDLSS-NR's reference WebGPU port, run unchanged on a three.js renderer.
//
// The code this entry point runs is OpenDLSS-NR by maan (https://github.com/maanHimself/OpenDLSS-NR,
// ports/browser-webgpu, MIT, Copyright (c) 2026 maan) at commit 9d08f41, vendored byte for byte under
// vendor/opendlss-nr/ (see its LICENSE and SOURCE.json). three-dlss-nr adds only the glue that puts it on three's
// GPUDevice behind the backend-neutral `NRBackend` interface, so it can be compared with the native TSL port.
// Not affiliated with NVIDIA. "DLSS" is an NVIDIA trademark used descriptively; no NVIDIA weights are included.
//
// A separate entry point so the main `three-dlss-nr` bundle does not carry the reference's ~200 KB of JS + WGSL.

export {
  loadReferenceModel,
  REFERENCE_REQUIREMENTS,
  REFERENCE_SOURCE,
  ReferenceWgslBackend,
  referenceUnavailableReason,
  referenceWgslBackend,
  sharedTensorBuffer,
  type ReferenceModel,
  type ReferenceWgslCreateOptions,
} from './ReferenceWgslBackend.js';

export {
  frameParams,
  ReferenceFrame,
  type FrameTexture,
  type ReferenceFrameInput,
  type ReferenceFrameOptions,
  type ReferenceFrameSettings,
} from './ReferenceFrame.js';
