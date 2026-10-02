// `three-dlss-nr/synthetic`: deterministic synthetic weights and input features, for tests and demos.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). No NVIDIA weights are included anywhere in this
// project: these weights are generated, in the reference's model directory layout, so that the network can be run and
// compared with the reference without them. A separate entry point, so the main bundle does not carry the generator.

export {
  generateSyntheticModel,
  generateSyntheticRecord,
  SYNTHETIC_FORMAT,
  type SyntheticModel,
  type SyntheticModelOptions,
} from './generate.js';
export {
  FEATURE_LANES,
  syntheticFeatures,
  syntheticProxy,
  centredProxy,
  type SyntheticFeaturesOptions,
} from './features.js';
export {
  SYNTHETIC_CONSTANTS,
  SYNTHETIC_F16_SIGMA,
  SYNTHETIC_GAINS,
  SYNTHETIC_RANGES,
  SYNTHETIC_ZERO_THRESHOLD,
} from './gains.js';
