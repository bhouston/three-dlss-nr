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

## Render pass

`DlssNrPass` puts a network behind a three.js scene: it renders the scene into an RGBA16F MRT target (linear colour
plus three's `velocity`), writes the network's input features from it on the GPU, runs the backend, composes the
result with the temporal history, and draws NR off, NR on or a split view onto the canvas (or the renderer's current
render target).

```ts
import { backendBuilder, createNRDevice, DlssNrPass } from 'three-dlss-nr';
import { loadReferenceModel, referenceWgslBackend } from 'three-dlss-nr/reference-backend';

const renderer = new WebGPURenderer({ device: (await createNRDevice()).device });
await renderer.init();
const pass = new DlssNrPass({ renderer, scene, camera, width: 960, height: 540, view: 'split' });
const model = await loadReferenceModel(renderer, modelFiles); // load once
await pass.setNetwork(backendBuilder(referenceWgslBackend, model)); // compiles; rebuilt by setSize()
pass.setSettings({ intensity: 1, style: 0 }); // the reference demo's NR controls
renderer.setAnimationLoop(async () => {
  const stats = await pass.render(); // null while the previous frame is still running
  // stats.network: { gpuMilliseconds, wallMilliseconds } of the network
});
pass.resetHistory(); // on a camera cut
```

- Both backends share the same frame kernels (`createFrameKernels`: ports of the reference's `input_features` and
  `compose`, byte-identical to `frame.wgsl` on D3D12), because they read the same features and write the same head.
  A builder may also return a backend with its own frame, such as `ReferenceFrame` (the upstream frame recorded into
  the upstream graph); the pass then draws its output.
- NR off is the reference's display transform (its ACES fit plus sRGB) of the scene, so both halves of the split go
  through the same display path. The pass sets no tone mapping and a linear output colour space while it draws.
- History restarts on the first frame, after `resetHistory()`, after a frame without NR, and after a network switch or
  rebuild. The internal size is capped at 1280x720 pixels.

## Contributing

How the port's kernels are written and tested against the reference (KernelBuilder, the numerics helpers, the
reference runner, and the TSL pitfalls): [src/README-internals.md](src/README-internals.md).

## License

MIT. The package's LICENSE carries both this project's copyright and OpenDLSS-NR's MIT notice; NOTICE carries
forward upstream's notice. Not affiliated with, endorsed by, or supported by NVIDIA Corporation. "DLSS" is a
trademark of NVIDIA Corporation, used here only descriptively.
