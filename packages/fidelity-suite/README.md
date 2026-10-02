# fidelity-suite

A side-by-side parity suite, built with [fidelity-kit](https://fidelity-kit.ben3d.ca), that shows
[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by [maan](https://github.com/maanHimself) (MIT, pinned at
`9d08f41`) and three-dlss-nr's three.js ports producing the same bytes. It is published on the demo site at
**[three-dlss-nr.ben3d.ca/parity/](https://three-dlss-nr.ben3d.ca/parity/)**. Private workspace package: none of
it reaches the library.

Each scene is rendered once by three.js and run through three renderers, fed the same input features and the same
synthetic weights:

| Renderer id          | What runs                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `opendlss-nr`        | The reference: upstream's WebGPU port standalone, its own device, WGSL and weight loader     |
| `three-dlss-nr-tsl`  | The native port: TSL compute nodes on a three.js `WebGPURenderer` (`tslBackend`)             |
| `three-dlss-nr-shim` | Upstream's WGSL vendored into the library, on the renderer's device (`referenceWgslBackend`) |

Scenes: the Lee Perry-Smith head scan (front and three-quarter views) and two procedural scenes (PBR spheres with an
HDR emissive ring; a checker room with thin rods), each at 256x256 and 512x512. Outputs: the composed output, the
input proxy, the head's RGB residual and blend weight, and four block boundaries (block 0, block 14, the ViT output at
block 38, block 69).

## What "parity" means here

- **Raw tensors, byte for byte.** For every scene, each renderer's input features, all 79 captured block boundaries,
  the three post tensors and the f32 head are read back and compared with the reference. Verdicts and SHA-256 digests
  go to `results/<size>/<scene>/exactness.json`.
- **Images, byte for byte.** The visualizations are fixed CPU functions of those tensors (`src/visualize.ts`) and
  the PNG encoder is deterministic (`scripts/png.mjs`), so equal tensors give byte-identical files. fidelity-kit then
  shows infinite PSNR and an empty delta for every pair.
- **Timing.** The same three renderers without boundary captures, interleaved frame by frame, timed with timestamp
  queries around the frame (`timing.json`, the scene READMEs and the summary table in `results/README.md`).

The weights are synthetic (`three-dlss-nr/synthetic`, seed 1), so the images are parity evidence, not pictures.

## Commands (repository root)

| Command                  | What it does                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `pnpm fidelity:generate` | Render every scene and size in real Chrome and rewrite `results/` (needs a GPU with hardware `shader-f16`) |
| `pnpm fidelity:check`    | Fail unless every committed verdict is bit-exact and every PNG on disk matches the reference's (no GPU)    |
| `pnpm fidelity:dev`      | Browse `results/` with `fidelity-kit dev` (processes metrics and deltas, live reload)                      |
| `pnpm fidelity:build`    | Export the static site into `packages/website/public/parity` (part of `pnpm build`)                        |

CI runs `pnpm fidelity:check` on the committed results. The website build exports the site from them, so deploying
needs no GPU.

## Regenerating the results

The reference needs `shader-f16`. Dawn in Node lacks it on Windows, and Mesa's lavapipe exposes it but folds the
reference's `f32(f16(x))` roundings away, so its results cannot be trusted. Regenerate in Chrome on real hardware
(the dev machine: RTX 3060 Ti, Chrome stable, D3D12 + DXC), on an otherwise idle GPU:

```sh
pnpm build                       # the page imports the built library; also generates the synthetic model once
pnpm fidelity:generate           # about 6 minutes on the dev machine; the TSL compiles dominate
pnpm fidelity:check
```

Options: `--sizes 256x256,512x512`, `--scenes head-front,spheres`, `--frames 10`, `--warmup 3`, `--model <dir>`,
`--chrome <path>`, `--headed`. A partial run rewrites only the scenes and sizes it covers, then rebuilds the results
READMEs from everything on disk (`pnpm --filter fidelity-suite readmes` does only that, without a GPU). The generator
uses the repository's DevTools harness
(`packages/three-dlss-nr/test/browser/chrome.mjs`): Chrome stable, else Playwright's Chromium.

Layout: `src/page.ts` is the browser page (scenes, the three renderers, comparisons, visualizations, timing),
`src/scenes.ts` the scenes, `src/features.ts` the CPU input features, `scripts/generate.mjs` the driver (serves the
page, the model and the reference's WGSL, receives the results), `scripts/check.mjs` the CI gate.

## Credits

[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by maan (MIT) is the reference. "Infinite, 3D Head Scan" by
Lee Perry-Smith ([Infinite-Realities](https://ir-ltd.net)), licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via the three.js examples. Not affiliated with NVIDIA;
"DLSS" is a trademark of NVIDIA Corporation, used descriptively. No NVIDIA weights are included.
