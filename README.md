# three-dlss-nr

[![ci](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-dlss-nr/actions/workflows/ci.yml)

> **A port to Three.js (TSL / WebGPU) of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
> [maan](https://github.com/maanHimself).**
>
> All credit for the network reimplementation, its numerics, and its documentation belongs to OpenDLSS-NR.
> This repository ports that work to three.js and tracks upstream commit
> [`9d08f41`](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41), included as the
> [`reference/OpenDLSS-NR`](reference/OpenDLSS-NR) submodule.

## What it is

[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) is an open-source reimplementation of the network used by
NVIDIA's DLSS 5 Neural Rendering (NR). It ships a Vulkan reference and a bit-exact browser WebGPU/WGSL port.

`three-dlss-nr` implements the same network with three.js
[TSL](https://threejs.org/docs/#api/en/nodes/TSL) compute kernels (`three/webgpu` + `three/tsl`), so it can run
inside a `WebGPURenderer` scene. The port is checked against the reference WebGPU implementation block by block on
the same inputs and weights. For speed and quality comparisons, the optional
`three-dlss-nr/reference-backend` entry point runs that reference WebGPU port itself, unchanged, on three.js' GPUDevice
behind the same backend interface (see the [package README](packages/three-dlss-nr#backends)).

This repository is a pnpm monorepo:

| Path                                               | Contents                                                      |
| -------------------------------------------------- | ------------------------------------------------------------- |
| [`packages/three-dlss-nr`](packages/three-dlss-nr) | The library (TypeScript, peer dependency `three`).            |
| [`packages/website`](packages/website)             | Demo site, deployed to Cloud Run.                             |
| [`reference/OpenDLSS-NR`](reference/OpenDLSS-NR)   | Upstream OpenDLSS-NR at `9d08f41` (git submodule, read-only). |

Status: bootstrap. The package currently exports placeholders while the network port is in progress.

## Weights

**You supply the weights. None are included.** The network's trained weights are NVIDIA's proprietary property
and are not part of this repository, its npm package, or OpenDLSS-NR. This project never downloads, extracts, or
redistributes them. To run the network on real weights, point it at a model directory you are entitled to use
(`manifest.json` + `model/stages/*`, in the layout described in the reference's
[`docs/weights.md`](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/weights.md)).

Tests use deterministic synthetic weights in the same layout and compare our TSL network against the reference
WebGPU port on them. Real-weight parity tests run only when `NR_WEIGHTS` / `NR_FIXTURES` point at local files.

## Development

```sh
git clone --recurse-submodules git@github.com:bhouston/three-dlss-nr.git
cd three-dlss-nr
pnpm install
pnpm build
pnpm test        # type check + unit tests (Node)
pnpm test:gpu    # WebGPU tests (headless Dawn in Node)
pnpm dev         # library watch + demo site
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and full list of checks.

## License and notices

MIT. See [LICENSE](LICENSE), which carries both this project's copyright (Ben Houston and contributors, 2026) and
OpenDLSS-NR's MIT notice (Copyright (c) 2026 maan), and [NOTICE](NOTICE), which carries forward upstream's notice.

This project is not affiliated with, endorsed by, or supported by NVIDIA Corporation or by the author of
OpenDLSS-NR. "DLSS" is a trademark of NVIDIA Corporation; it is used here only to describe what the network
implemented by this code is compatible with. No NVIDIA software, weights, headers, or documentation are included,
and no rights under any NVIDIA intellectual property are granted.
