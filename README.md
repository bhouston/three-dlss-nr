# three-dlss-nr

[![ci](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml)
[![Unit coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbhouston%2Fthree-dlss-nr%2Fcoverage-badge%2Fcoverage.json)](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bhouston/three-dlss-nr/blob/main/LICENSE)
[![Live demo](https://img.shields.io/badge/demo-three--dlss--nr.ben3d.ca-blue)](https://three-dlss-nr.ben3d.ca)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/5J5Ur3F6Z2)

> **A port to Three.js (TSL / WebGPU) of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
> [maan](https://github.com/maanHimself).**
>
> OpenDLSS-NR is an open-source reimplementation of the network behind NVIDIA's DLSS 5 Neural Rendering. All credit
> for that reimplementation, its numerics and its documentation belongs to OpenDLSS-NR. This repository ports it to
> three.js and tracks upstream commit [`9d08f41`](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41), included
> as the [`reference/OpenDLSS-NR`](reference/OpenDLSS-NR) submodule. Not affiliated with NVIDIA, and no NVIDIA weights
> are included ([details](#license-and-notices)).

**The DLSS 5 neural rendering network, as three.js compute nodes, byte for byte.** three-dlss-nr runs the 71-block
OpenDLSS-NR network as [TSL](https://threejs.org/docs/#api/en/nodes/TSL) compute kernels inside a `WebGPURenderer`,
fed straight from your three.js scene with no CPU readback. Every intermediate tensor is byte-identical to the
reference WebGPU port.

![The demo in split view: the Lee Perry-Smith head scan rendered normally on the left and through the TSL network on the right](docs/images/demo-tsl-synthetic-split.png)

_**Synthetic weights: output is not meaningful.** The demo in split view on the TSL backend, NR off on the left and
NR on on the right. Synthetic weights have the right layout and run the network's real arithmetic, but the image they
produce means nothing, hence the red field ([why](#why-there-are-no-real-weights-and-what-the-synthetic-ones-are-for)).
With a real model directory the right half is the re-rendered head. Head: Lee Perry-Smith, CC BY 3.0
([credits](#credits))._

Try it live at **[three-dlss-nr.ben3d.ca](https://three-dlss-nr.ben3d.ca)**.

## What the network does

The website now opens with a [Hungarian interactive explainer](http://localhost:3300/) for presentations:
DLSS 5 in plain language, live Three.js objects, scroll parallax, accessible draggable comparisons using real
local demo captures, and a screenshot gallery. The original network demo is at `/demo`; `/local-demo.html`
retains the configured local-model workflow. On this Windows setup, double-click `start-presentation.cmd`.
See [the explainer guide](docs/bemutato.md) and [local startup instructions](INDITAS.md).

From [upstream's README](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41#the-network): it is a generative
neural rendering network (NVIDIA's term). It **re-renders the frame the engine already drew**, generating detail from
injected noise and adjusting tone, structure and skin under a style setting. Input and output are the same
resolution; **it is not an upscaler**.

- A U-net of shifted-window (Swin) transformer blocks with a global ViT at the bottom: 71 blocks over six pooling
  levels, FP8 (E4M3) activations with FP16 accumulation, 141 MiB of weights.
- In: one rendered frame (a low dynamic range proxy of it, three lanes of Gaussian noise, the previous frame's output
  reprojected, and five conditioning scalars). Out: four f32 channels per pixel, an RGB residual and one
  temporal-blend logit.
- 451 compute dispatches per frame in the WebGPU formulation.

NVIDIA describes the model in its report,
[DLSS 5: Generative Neural Rendering](https://research.nvidia.com/labs/adlr/DLSS5/files/DLSS5_Report.pdf)
([project page](https://research.nvidia.com/labs/adlr/DLSS5/)). Upstream's
[`docs/network.md`](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/network.md) is the graph in full.

## Verified byte for byte

The spec for every kernel is upstream's WebGPU port (`ports/browser-webgpu`), and the port is checked against it on
the same device, the same inputs and the same weights:

- **Every kernel family** (FP8 and f16 GEMMs, window attention, the global ViT, the elementwise ops, preprocess, and
  the frame kernels that build the input features and compose the output) is byte-identical to the reference.
- **The whole network** (451 dispatches) is byte-identical at 64x64 and at 512x512 on synthetic weights: all 79 block
  boundaries, the three post tensors and the head, 83 of 83 tensors, with repeat runs identical. Checked in real
  Chrome on an RTX 3060 Ti (D3D12 + DXC), with both networks in one page on one device.
- **The numerics** (E4M3 encode and decode, f16 rounding, the SiLU and exponent tables, FP8 and f16 dot products) are
  checked on the GPU against all seven sections of the reference's numerics fixture, which upstream's Vulkan
  implementation produced: exhaustively over every half and every byte where the domain allows it.
- **CI** (Linux, Mesa lavapipe) runs the whole network in Node and gates it against golden digests recorded from the
  Chrome run. It cannot compare against the reference directly: llvmpipe folds the `f32(f16(x))` round trip into `x`,
  so the reference's f16 roundings silently do not happen there. The TSL port never relies on f16 hardware. It rounds
  on f32 bit patterns, so it produces the same bytes on lavapipe as on a real GPU.

**See it side by side:** [three-dlss-nr.ben3d.ca/parity/](https://three-dlss-nr.ben3d.ca/parity/) shows the
reference running standalone, the TSL port and the reference shim on the same frames (the head scan and two
procedural scenes, at 256x256 and 512x512): the composed output, the head and four block boundaries, all byte-identical,
with every raw tensor compared and per-scene GPU times. It is built with [fidelity-kit](https://fidelity-kit.ben3d.ca)
from [`packages/fidelity-suite`](packages/fidelity-suite), and CI checks its committed verdicts.

A consequence: **the TSL port does not need `shader-f16`**, while the reference does. How the kernels are written and
tested is in [`src/README-internals.md`](packages/three-dlss-nr/src/README-internals.md).

## Usage

```sh
npm install three-dlss-nr three
```

`DlssNrPass` puts the network behind a three.js scene. It renders the scene into an RGBA16F target (linear colour
plus three's `velocity`), builds the network's input features on the GPU, runs the network, composes the result with
the temporal history, and draws NR off, NR on or a split view onto the canvas.

```ts
import { backendBuilder, createNRRenderer, DlssNrPass, NRModel, tslBackend } from 'three-dlss-nr';

const { renderer } = await createNRRenderer(); // a WebGPURenderer on a device with the limits the network needs
document.body.appendChild(renderer.domElement);

// Your model directory: manifest.json + model/stages/* (see the weights section below).
const model = await NRModel.load('https://example.com/my-model');

const pass = new DlssNrPass({ renderer, scene, camera, width: 960, height: 540, view: 'split' });
await pass.setNetwork(backendBuilder(tslBackend, model)); // compiles the kernels; setSize() rebuilds
pass.setSettings({ intensity: 1, style: 1 }); // style: 0 none, 1 cinematic, 2 natural

renderer.setAnimationLoop(async () => {
  const stats = await pass.render(); // null while the previous frame is still on the GPU
  // stats?.network: { method, gpuMilliseconds, wallMilliseconds }
});
```

Load the model once and share it: resizes and network rebuilds reuse it instead of re-reading 141 MiB. Call
`pass.resetHistory()` on a camera cut. The internal size is capped at 1280x720.

To run the network on your own inputs instead of a scene, use a backend directly:

```ts
const network = await tslBackend.create({ renderer, model, width: 512, height: 512, onProgress: console.log });
network.writeFeatures(features); // Float32Array, fullRows x 16 lanes (or write network.features from a compute pass)
const timing = await network.run({ timing: true });
const head = await network.readHead(); // Float32Array, fullRows x 4: RGB residual + blend logit
network.dispose();
```

See the [package README](packages/three-dlss-nr/README.md) for the full API.

## Backends

The network runs behind one interface, `NRBackend`, with two implementations that read the same features and write
the same head, so they can be switched at runtime and compared on identical inputs:

| Factory                | Entry point                       | What runs                                                                                          | Needs                                  |
| ---------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `tslBackend`           | `three-dlss-nr`                   | The native port: three.js TSL compute nodes, one `renderer.compute()` per frame                    | WebGPU, 24 KiB workgroup storage       |
| `referenceWgslBackend` | `three-dlss-nr/reference-backend` | **Upstream's own WebGPU port by maan, unchanged** (its JS and WGSL at `9d08f41`) on three's device | `shader-f16`, 32 KiB workgroup storage |

The reference backend exists to compare speed and output against the original on the same renderer. Its code is
OpenDLSS-NR's (MIT, Copyright (c) 2026 maan), copied byte for byte at build time with upstream's LICENSE and a
SHA-256 provenance list. It is a separate entry point, so the main bundle does not carry it. Each factory's
`unavailableReason(renderer)` returns `null` or a sentence naming what is missing and how to fix it.

A third entry point, `three-dlss-nr/synthetic`, generates deterministic synthetic weights in the model directory
layout (`generateSyntheticModel`) and synthetic input features (`syntheticFeatures`), for tests and demos.

## Why there are no real weights (and what the synthetic ones are for)

**You supply the weights. None are included, downloaded, hosted or extracted.** The trained weights are NVIDIA's
proprietary DLSS-NR 310.8.0 model. OpenDLSS-NR deliberately ships neither the weights nor instructions for obtaining
them, and grants no rights under NVIDIA's intellectual property. This project cannot redistribute, host or extract
them either: pulling them out of NVIDIA's binaries would likely breach NVIDIA's license terms. So neither this
repository, the npm package nor the demo contains them.

**Synthetic weights** (`three-dlss-nr/synthetic`) are generated deterministically, in the same model directory layout,
and calibrated so that every block boundary stays in a realistic, non-saturating range. They serve two purposes:

- **They prove the port computes the same function as the reference.** The port is bit-exact against the reference
  on arbitrary weights, through every kernel and every block, so it will match on any weights, including the real
  ones.
- **They let the whole pipeline run end to end** in the tests and in the demo.

Their output image is meaningless (the red field in the screenshot above).

**If you are entitled to real weights,** point the network at the model directory: `manifest.json` plus
`model/stages/*`, in the layout described in upstream's
[`docs/weights.md`](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/weights.md). In the demo, use
**Load model directory...**; the browser reads it locally and never uploads it. In code, pass its URL or files to
`NRModel.load`. The real-capture parity test (`network.real.gpu.test.ts`) runs upstream's own fixture comparison when
`NR_WEIGHTS` (a model directory) and `NR_FIXTURES` (a fixture directory) point at local files.

There are two paths to meaningful public output: a license from NVIDIA, or open weights trained for the same
architecture. See [docs/training-weights.md](docs/training-weights.md).

## Demo

**[three-dlss-nr.ben3d.ca](https://three-dlss-nr.ben3d.ca)** runs the network live in the browser on the
Lee Perry-Smith head scan. It needs a WebGPU browser (current Chrome or Edge, or Safari 26+).

![The demo with NR off: the head scan with the view, weights, backend, NR settings, scene and device panels](docs/images/demo-nr-off.png)

- **View:** NR off, NR on, or a split view with a draggable divider.
- **Weights:** load your own model directory (read by the browser, never uploaded) or generate synthetic weights.
- **Backend:** TSL (native three.js) or Reference WGSL (OpenDLSS-NR), with the network's GPU time per frame.
- **NR settings:** intensity, style (none, cinematic, natural), local tone and structure, skin structure, auto mask,
  colour strength and paper white, as in upstream's demo.
- **Scene and device:** resolution up to 1280x720, and a report of the adapter's features and limits.

The site is [`packages/website`](packages/website) (TanStack Start, React, Tailwind), deployed to Cloud Run from
`main`. Run it locally with `pnpm dev`.

## Performance

**Not real-time today.** The whole network takes hundreds of milliseconds per frame on an RTX 3060 Ti. The native
TSL port is currently about 1.6x slower than the reference's hand-tuned WGSL on the same device, with the same bytes
out. Closing that gap is future work.

TSL vs reference WGSL, whole network, ms per frame: median GPU time (minimum in parentheses). RTX 3060 Ti, Chrome
stable (D3D12 + DXC), synthetic weights, 451 dispatches per frame, 30 frames after 5 warm-up frames, timed with
timestamp queries around the frame (wall time is within about 1.5 ms of GPU time).

| Backend                                 | 512x512 (field 576x512) | 1280x720 (field 1344x768) |
| --------------------------------------- | ----------------------: | ------------------------: |
| TSL (`tslBackend`)                      |           201.2 (199.5) |             658.6 (655.2) |
| Reference WGSL (`referenceWgslBackend`) |           123.4 (122.6) |             412.2 (410.8) |

|                                     | TSL            | Reference WGSL |
| ----------------------------------- | -------------- | -------------- |
| Network creation (pipeline compile) | about 35 s     | about 17 s     |
| First frame, 512x512 / 1280x720     | 420 / 845 ms   | 165 / 648 ms   |
| Weights on the GPU                  | about 141 MiB  | about 144 MiB  |
| Activations, 512x512 / 1280x720     | 545 / 1905 MiB | 545 / 1905 MiB |

For context, upstream reports 72-73 ms at 512x512 for its WebGPU port on an RTX 4070 SUPER, and 2.8 ms at 768x768 and
7.8 ms at 1920x1080 for its Vulkan FP8 tensor-core path on the same card. Those are upstream's numbers, measured on
different hardware.

Reproduce on an otherwise idle GPU (the script runs both backends on one renderer in real Chrome, feeds them the same
features and times them the same way):

```sh
pnpm build && node scripts/bench-backends.mjs --sizes 512x512,1280x720
```

## Requirements

- three.js r180 or later (`three` is a peer dependency; developed and tested on 0.186) and its `WebGPURenderer`. There
  is no WebGL fallback.
- A WebGPU device with at least 24 KiB of workgroup storage, 256 invocations per workgroup and 8 storage buffers per
  shader stage. `createNRDevice()` and `createNRRenderer()` request the adapter's maximum limits plus `shader-f16` and
  `timestamp-query` when available.
- The reference backend also needs `shader-f16` and 32 KiB of workgroup storage.

## Repository layout

| Path                                                                                               | Contents                                                                                                                  |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [`packages/three-dlss-nr`](packages/three-dlss-nr)                                                 | The library, published to npm. See its [README](packages/three-dlss-nr/README.md).                                        |
| [`packages/three-dlss-nr/src/README-internals.md`](packages/three-dlss-nr/src/README-internals.md) | How kernels are written in TSL and tested against the reference, and the TSL pitfalls.                                    |
| [`packages/website`](packages/website)                                                             | The demo site, deployed to Cloud Run.                                                                                     |
| [`packages/fidelity-suite`](packages/fidelity-suite)                                               | The side-by-side parity suite (fidelity-kit) and its committed results, served at `/parity/`.                             |
| [`reference/OpenDLSS-NR`](reference/OpenDLSS-NR)                                                   | Upstream OpenDLSS-NR at `9d08f41` (git submodule, read-only), used for parity tests and bundled by the reference backend. |
| [`docs/`](docs)                                                                                    | Screenshots and [candidate head assets](docs/suggested-assets.md) for the demo, with their licenses.                      |
| [`scripts/`](scripts)                                                                              | Benchmark, bundle-size gate, release and CI helpers.                                                                      |

## Development

```sh
git clone --recurse-submodules https://github.com/bhouston/three-dlss-nr.git
cd three-dlss-nr
corepack enable
pnpm install
pnpm build
pnpm dev           # library watch + demo site at http://localhost:3300
```

| Command               | What it does                                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `pnpm build`          | Bundle the reference port into `vendor/`, then build the library and the website                                      |
| `pnpm tsc`            | Type-check the workspace                                                                                              |
| `pnpm lint`           | Oxlint                                                                                                                |
| `pnpm format`         | Oxfmt (`pnpm format:check` in CI)                                                                                     |
| `pnpm test`           | Type-check, then the unit tests (Node) with coverage                                                                  |
| `pnpm test:gpu`       | The WebGPU tests in Node: the `gpu` project, then the whole network in the `gpu-network` project                      |
| `pnpm size`           | Minified gzip bundle-size gate, `three` external: 40 kB for `three-dlss-nr`, 8 kB for `three-dlss-nr/synthetic`       |
| `pnpm bench:backends` | Benchmark TSL against reference WGSL in real Chrome (`--sizes 512x512,1280x720`, `--model <dir>`)                     |
| `pnpm fidelity:check` | Check the committed parity results (every tensor and image bit-exact); `fidelity:generate` regenerates them in Chrome |
| `pnpm fidelity:dev`   | Browse the parity results locally with fidelity-kit                                                                   |

**GPU tests in Node.** `pnpm test:gpu` runs the `*.gpu.test.ts` files on headless WebGPU (Google's Dawn, via
[`vitest-environment-webgpu-node`](https://github.com/bhouston/vitest-gpu)), with no browser. Dawn uses the platform
backend (D3D12 on Windows, Metal on macOS, Vulkan on Linux); set `DLSS_NR_DAWN_BACKEND=vulkan` or `DLSS_NR_DAWN` to
choose another. Linux without a GPU needs Mesa's software Vulkan driver, as in CI:

```sh
sudo apt-get install -y libegl1 libgles2 libgl1-mesa-dri mesa-vulkan-drivers
export LIBGL_ALWAYS_SOFTWARE=1
```

The whole-network tests run at 64x64 by default. `NR_FULL=1` adds 512x512, whose single 451-dispatch command buffer
can trip the Windows driver timeout (TDR) when the GPU is busy. On Windows, Dawn's Vulkan backend loads only when
`vulkan-1.dll` sits beside `node.exe` or `dawn.node`.

**Parity in real Chrome.** Dawn in Node has no `shader-f16` on Windows, so reference parity runs in Chrome (stable,
else Playwright's Chromium) through a small DevTools harness:

```sh
node packages/three-dlss-nr/test/browser/run-network-parity-chrome.mjs --sizes 64x64,512x512   # whole network, --golden re-pins CI digests
node packages/three-dlss-nr/test/browser/run-shim-parity-chrome.mjs                            # reference backend vs standalone upstream
node packages/three-dlss-nr/test/browser/run-fp8-gemm-chrome.mjs                               # FP8 GEMM against the CPU oracle
pnpm bench:backends                                                                            # TSL vs reference WGSL timings
pnpm fidelity:generate                                                                         # the side-by-side parity results (packages/fidelity-suite)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue, branch, PR and commit workflow, the full list of local checks and
the release process, and [SECURITY.md](SECURITY.md) for private vulnerability reports. Treat `reference/OpenDLSS-NR`
as read-only, and never download, extract or commit NVIDIA model weights.

## License and notices

MIT. [LICENSE](LICENSE) carries both this project's copyright (Ben Houston and contributors, 2026) and OpenDLSS-NR's
MIT notice (Copyright (c) 2026 maan). [NOTICE](NOTICE) carries forward upstream's notice and lists the redistributed
reference code and demo assets.

This project is not affiliated with, endorsed by, or supported by NVIDIA Corporation or by the author of
OpenDLSS-NR. "DLSS" is a trademark of NVIDIA Corporation; it is used here only to describe what the network
implemented by this code is compatible with. No NVIDIA software, weights, headers, or documentation are included,
and no rights under any NVIDIA intellectual property are granted. You are responsible for the licenses that apply to
whatever model data you use with it.

## Credits

- [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by [maan](https://github.com/maanHimself) (MIT): the
  reimplementation of the network, its numerics, its documentation, and the WebGPU port that is this project's spec
  and its reference backend.
- "Infinite, 3D Head Scan" by Lee Perry-Smith ([Infinite-Realities](https://ir-ltd.net)), licensed under
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/): the demo's head. glTF and maps via the
  [three.js examples](https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/LeePerrySmith).
- [three.js](https://threejs.org) and its TSL node system, which the port is written in.

## Author and sponsor

Created by [Ben Houston](https://ben3d.ca) and sponsored by [Land of Assets](https://landofassets.com).
