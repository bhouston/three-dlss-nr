# three-dlss-nr

[![npm version](https://img.shields.io/npm/v/three-dlss-nr.svg)](https://www.npmjs.com/package/three-dlss-nr)
[![npm downloads](https://img.shields.io/npm/dm/three-dlss-nr.svg)](https://www.npmjs.com/package/three-dlss-nr)
[![ci](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bhouston/three-dlss-nr/blob/main/LICENSE)
[![Live demo](https://img.shields.io/badge/demo-three--dlss--nr.ben3d.ca-blue)](https://three-dlss-nr.ben3d.ca)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/5J5Ur3F6Z2)

> **A port to Three.js (TSL / WebGPU) of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
> [maan](https://github.com/maanHimself)** (MIT, upstream commit
> [`9d08f41`](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41)), an open-source reimplementation of the
> network behind NVIDIA's DLSS 5 Neural Rendering. Not affiliated with NVIDIA; no NVIDIA weights are included.

The DLSS 5 neural rendering network as three.js [TSL](https://threejs.org/docs/#api/en/nodes/TSL) compute kernels,
running inside a `WebGPURenderer` and fed straight from your scene. Every intermediate tensor is byte-identical to the
reference WebGPU port.

![The demo in split view: the head scan rendered normally on the left and through the TSL network with synthetic weights on the right](https://raw.githubusercontent.com/bhouston/three-dlss-nr/main/docs/images/demo-tsl-synthetic-split.png)

_The demo with NR off (left) and on (right), on **synthetic weights**, which produce meaningless output. With a real
model directory the right half is the re-rendered head._

The network re-renders the frame the engine already drew at the same resolution, generating detail from injected
noise and adjusting tone, structure and skin under a style setting. It is not an upscaler. It is a 71-block U-net of
shifted-window (Swin) transformer blocks with a global ViT at the bottom, FP8 (E4M3) activations with FP16
accumulation, 451 compute dispatches per frame. NVIDIA describes the model in
[DLSS 5: Generative Neural Rendering](https://research.nvidia.com/labs/adlr/DLSS5/files/DLSS5_Report.pdf).

Try it live at **[three-dlss-nr.ben3d.ca](https://three-dlss-nr.ben3d.ca)**.

## Install

```bash
npm install three-dlss-nr three
```

## Weights

**You supply the weights; none are included, downloaded, hosted or extracted.** The trained weights are NVIDIA's
proprietary DLSS-NR 310.8.0 model. Neither this package nor OpenDLSS-NR distributes them or explains how to obtain
them, and extracting them from NVIDIA's binaries would likely breach NVIDIA's license terms. If you are entitled to a
model directory, pass it to `NRModel.load`: `manifest.json` plus `model/stages/*`, in the layout described in
upstream's [`docs/weights.md`](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/weights.md).

`three-dlss-nr/synthetic` generates deterministic weights in the same layout, calibrated to keep every block in a
realistic range. Because the port is bit-exact against the reference on these arbitrary weights, it will match on any
weights, real ones included. Synthetic weights let the pipeline run end to end, but the image they produce is
meaningless. The
[repository README](https://github.com/bhouston/three-dlss-nr#why-there-are-no-real-weights-and-what-the-synthetic-ones-are-for)
explains why and the paths to meaningful output (a license from NVIDIA, or
[open weights trained for the same architecture](https://github.com/bhouston/three-dlss-nr/blob/main/docs/training-weights.md)).

## Quick start

`DlssNrPass` renders a three.js scene into an RGBA16F target (linear colour plus three's `velocity`), builds the
network's input features on the GPU, runs the network, composes the result with the temporal history, and draws NR
off, NR on or a split view onto the canvas.

```ts
import { backendBuilder, createNRRenderer, DlssNrPass, NRModel, tslBackend } from 'three-dlss-nr';

const { renderer } = await createNRRenderer(); // a WebGPURenderer on a device with the limits the network needs
document.body.appendChild(renderer.domElement);

const model = await NRModel.load('https://example.com/my-model'); // load once, share across resizes

const pass = new DlssNrPass({ renderer, scene, camera, width: 960, height: 540, view: 'split' });
await pass.setNetwork(backendBuilder(tslBackend, model)); // compiles the kernels; setSize() rebuilds
pass.setSettings({ intensity: 1, style: 1 }); // style: 0 none, 1 cinematic, 2 natural

renderer.setAnimationLoop(async () => {
  const stats = await pass.render(); // null while the previous frame is still on the GPU
  // stats?.network: { method, gpuMilliseconds, wallMilliseconds }
});

pass.resetHistory(); // on a camera cut
```

- `view` is `'off'`, `'on'` or `'split'` (with `pass.split` in [0, 1]); change it at any time.
- `setSettings` takes the reference demo's controls: `intensity`, `style`, `localTone`, `localStructure`,
  `skinStructure`, `autoMask`, `paperWhite`, `colorStrength` (defaults in `DLSS_NR_DEFAULT_SETTINGS`).
- NR off is the reference's display transform (its ACES fit plus sRGB) of the scene, so both halves of the split go
  through the same display path. The pass sets no tone mapping and a linear output colour space while it draws.
- History restarts on the first frame, after `resetHistory()`, after a frame without NR, and after a network switch or
  rebuild. The internal size is capped at 1280x720 (`DLSS_NR_MAX_PIXELS`).
- `pass.renderTarget` (the rendered colour and velocity) and `pass.outputTexture` (the presented NR image) are exposed
  for further processing.

## Running the network directly

Every backend implements `NRBackend`. Its input and output are GPU-resident tensors, so your own compute passes can
write the features and read the head without a CPU round trip.

```ts
import { createNRRenderer, NRModel, tslBackend } from 'three-dlss-nr';
import { generateSyntheticModel, syntheticFeatures } from 'three-dlss-nr/synthetic';

const { renderer } = await createNRRenderer();
const model = await NRModel.load(await generateSyntheticModel({ seed: 1 })); // or a model directory URL

const network = await tslBackend.create({ renderer, model, width: 512, height: 512, onProgress: console.log });
network.writeFeatures(syntheticFeatures(network.geometry)); // Float32Array, fullRows x 16 (or write network.features)
const timing = await network.run({ timing: true }); // { method, gpuMilliseconds, wallMilliseconds }
const head = await network.readHead(); // Float32Array, fullRows x 4: RGB residual + blend logit
network.dispose(); // a model passed in is borrowed: dispose it yourself
```

Pass `captureBoundaries: true` to `create` to read any of the 79 block boundaries afterwards with
`readBoundary('block-12')`; `readTensor(label)` reads any intermediate by the reference's label.

## Backends

| Factory                | Entry point                       | What runs                                                                                             | Needs                                  |
| ---------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `tslBackend`           | `three-dlss-nr`                   | The native port: three.js TSL compute nodes, one `renderer.compute()` per frame                       | WebGPU, 24 KiB workgroup storage       |
| `referenceWgslBackend` | `three-dlss-nr/reference-backend` | **OpenDLSS-NR's own WebGPU port by maan, unchanged** (its JS and WGSL at `9d08f41`) on three's device | `shader-f16`, 32 KiB workgroup storage |

Both read the same features and write the same head, so `DlssNrPass` and your own frame code work with either, and you
can switch at runtime and compare speed and output on identical inputs. `factory.unavailableReason(renderer)` returns
`null` or a sentence naming what the device lacks and how to fix it.

The `reference-wgsl` backend **is the upstream code** (MIT, Copyright (c) 2026 maan), vendored byte for byte with
upstream's LICENSE and a SHA-256 provenance list (`vendor/opendlss-nr/`). This package only puts it on a three.js
device behind the same interface. It is a separate entry point so the main bundle does not carry it:

```ts
import { backendBuilder, createNRDevice, DlssNrPass } from 'three-dlss-nr';
import { loadReferenceModel, referenceWgslBackend } from 'three-dlss-nr/reference-backend';
import { WebGPURenderer } from 'three/webgpu';

// createNRDevice requests shader-f16 and the adapter's maximum limits.
const renderer = new WebGPURenderer({ device: (await createNRDevice()).device });
await renderer.init();

const reason = referenceWgslBackend.unavailableReason(renderer); // null, or why and how to fix it
const model = await loadReferenceModel(renderer, 'https://example.com/my-model'); // once; share across resizes
const pass = new DlssNrPass({ renderer, scene, camera, width: 960, height: 540 });
await pass.setNetwork(backendBuilder(referenceWgslBackend, model));
```

`ReferenceFrame` (same entry point) adds the upstream demo's own frame kernels (input features, compose, history),
fed from a three.js render target with no CPU readback, for comparing the whole upstream frame as is.

## API overview

| Export                                                     | Purpose                                                                                             |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `DlssNrPass`, `backendBuilder`, `DLSS_NR_DEFAULT_SETTINGS` | Scene to network to canvas, with NR off / on / split                                                |
| `tslBackend`, `NRNetwork`                                  | The native TSL network (`NRBackendFactory` and its backend class)                                   |
| `NRBackend`, `NRBackendFactory`, `NRFrameTiming`           | The backend-neutral interface both backends implement                                               |
| `NRModel`, `parseManifest`, `validateManifest`             | Load and validate a model directory (URL or in memory)                                              |
| `createNRDevice`, `createNRRenderer`, `nrDeviceProblems`   | A device and renderer configured for the network, and why a device cannot run it                    |
| `createFrameKernels`, `NRHistory`                          | The frame kernels: input features from a three.js render, compose with history                      |
| `NRFrameTimer`, `summarizeMilliseconds`                    | Timestamp-query timing, the same for both backends                                                  |
| `NRGraph`, `runKernels`, `kernelWGSL`                      | The 451-dispatch graph and kernel helpers, for parity and debugging work                            |
| `three-dlss-nr/synthetic`                                  | `generateSyntheticModel`, `syntheticFeatures`: deterministic weights and inputs for tests and demos |
| `three-dlss-nr/reference-backend`                          | `referenceWgslBackend`, `loadReferenceModel`, `ReferenceFrame`: upstream's WebGPU port              |

## Requirements

- three.js r180 or later (peer dependency; developed and tested on 0.186) and its `WebGPURenderer`. There is no
  WebGL fallback.
- A WebGPU device with at least 24 KiB of workgroup storage, 256 invocations per workgroup and 8 storage buffers per
  shader stage. The TSL backend does **not** need `shader-f16`: it rounds halves on f32 bit patterns, so it gives the
  same bytes on any device.
- The reference backend also needs `shader-f16`, 32 KiB of workgroup storage and 512 invocations per workgroup.

## Performance

Not real-time today. On an RTX 3060 Ti in Chrome, the whole network takes a median 201 ms per frame at 512x512 and
659 ms at 1280x720 on the TSL backend, against 123 ms and 412 ms for the reference WGSL backend on the same device.
Closing that gap is future work. Full numbers and how to reproduce them are in the
[repository README](https://github.com/bhouston/three-dlss-nr#performance).

## Verification

Every kernel family and the whole network (451 dispatches: all 79 block boundaries, the three post tensors and the
head) are byte-identical to the reference WebGPU port at 64x64 and 512x512 on synthetic weights, checked in real
Chrome on an RTX 3060 Ti. The numerics are checked against the reference's numerics fixture, exhaustively over every
half and every byte where the domain allows it. CI gates the whole network against golden digests. See the
[repository README](https://github.com/bhouston/three-dlss-nr#verified-byte-for-byte) and the
[internals notes](https://github.com/bhouston/three-dlss-nr/blob/main/packages/three-dlss-nr/src/README-internals.md).

## License

MIT. The package's LICENSE carries both this project's copyright and OpenDLSS-NR's MIT notice (Copyright (c) 2026
maan); NOTICE carries forward upstream's notice. The `reference-backend` entry point redistributes upstream's
MIT-licensed WebGPU port unmodified, with its LICENSE.

Not affiliated with, endorsed by, or supported by NVIDIA Corporation or by the author of OpenDLSS-NR. "DLSS" is a
trademark of NVIDIA Corporation, used here only to describe what the network implemented by this code is compatible
with. No NVIDIA software, weights, headers, or documentation are included, and no rights under any NVIDIA
intellectual property are granted.

## Credits

[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by [maan](https://github.com/maanHimself): the
reimplementation of the network, its numerics and its documentation, and the WebGPU port this package is checked
against and redistributes as its reference backend.

## Author and sponsor

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
