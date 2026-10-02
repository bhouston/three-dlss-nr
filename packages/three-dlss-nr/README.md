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

## Backends

The network runs behind one interface, `NRBackend` (exported from `three-dlss-nr`), with two implementations:

| id               | what runs                                                                                                           | needs                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `tsl`            | the native port: three.js TSL compute nodes (in progress)                                                           | WebGPU, 24 KiB workgroup storage       |
| `reference-wgsl` | **OpenDLSS-NR's own WebGPU port by maan, unchanged** (its JS and WGSL at `9d08f41`, vendored), on three's GPUDevice | `shader-f16`, 32 KiB workgroup storage |

The `reference-wgsl` backend **is the upstream code** (MIT, Copyright (c) 2026 maan); this package only puts it on a
three.js device, behind the same interface, so the two can be compared for speed and output. It is a separate entry
point so the main bundle does not carry it:

```ts
import { createNRDevice } from 'three-dlss-nr';
import { loadReferenceModel, referenceWgslBackend } from 'three-dlss-nr/reference-backend';
import { WebGPURenderer } from 'three/webgpu';

// A device with 'shader-f16' and raised limits. (WebGPURenderer creating its own device requests every feature the
// adapter has, but only default limits: then pass requiredLimits: { maxComputeWorkgroupStorageSize: 32768, ... }.)
const renderer = new WebGPURenderer({ device: (await createNRDevice()).device });
await renderer.init();

const reason = referenceWgslBackend.unavailableReason(renderer); // null, or why and how to fix it
const model = await loadReferenceModel(renderer, 'https://example.com/my-model'); // once; share across resizes
const backend = await referenceWgslBackend.create({ renderer, model, width: 1280, height: 720 });
backend.writeFeatures(features); // or write backend.features (a storage attribute) from a compute pass
const timing = await backend.run({ timing: true }); // { method, gpuMilliseconds, wallMilliseconds }
const head = await backend.readHead(); // or bind backend.head in a compose pass
```

`ReferenceFrame` (same entry point) adds the upstream demo's own frame kernels (input features, compose, history),
fed straight from a three.js render target (RGBA16F colour plus the `velocity` MRT output), with no CPU readback.

## Contributing

How the port's kernels are written and tested against the reference (KernelBuilder, the numerics helpers, the
reference runner, and the TSL pitfalls): [src/README-internals.md](src/README-internals.md).

## License

MIT. The package's LICENSE carries both this project's copyright and OpenDLSS-NR's MIT notice; NOTICE carries
forward upstream's notice. Not affiliated with, endorsed by, or supported by NVIDIA Corporation. "DLSS" is a
trademark of NVIDIA Corporation, used here only descriptively.
