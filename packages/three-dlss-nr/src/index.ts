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
