# three-dlss-nr

[![npm version](https://img.shields.io/npm/v/three-dlss-nr.svg)](https://www.npmjs.com/package/three-dlss-nr)
[![ci](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml)

**A port to three.js (TSL / WebGPU) of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
[maan](https://github.com/maanHimself)** (MIT, upstream commit
[`9d08f41`](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41)), an open-source reimplementation of the
network used by NVIDIA's DLSS 5 Neural Rendering. The network runs as three.js TSL compute kernels inside a
`WebGPURenderer`.

Status: bootstrap. The package currently exports placeholders while the network port is in progress.

## Install

```bash
npm install three-dlss-nr three
```

## Weights

You supply the weights; none are included. NVIDIA's trained weights are proprietary and are not distributed by
this package or by OpenDLSS-NR. See the [repository README](https://github.com/bhouston/three-dlss-nr#weights).

## Contributing

How the port's kernels are written and tested against the reference (KernelBuilder, the numerics helpers, the
reference runner, and the TSL pitfalls): [src/README-internals.md](src/README-internals.md).

## License

MIT. The package's LICENSE carries both this project's copyright and OpenDLSS-NR's MIT notice; NOTICE carries
forward upstream's notice. Not affiliated with, endorsed by, or supported by NVIDIA Corporation. "DLSS" is a
trademark of NVIDIA Corporation, used here only descriptively.
