# Demo scene assets

The original Lee Perry-Smith head remains the default. Four optional subjects offer different shapes and embedded
PBR materials without changing the homepage, backend defaults or local/synthetic weight-loading flow.

All new subjects come independently from [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
at commit `edc7c9e67c639d230715049ee31f9a96a6babbbe`. No binaries from PR #4 were used.
`packages/website/demo-model-sources.json` records exact pinned source URLs, source SHA-256 values, output hashes,
byte sizes and modification notes. Each distributed directory contains the exact upstream `SOURCE-LICENSE.txt`.
The registry displays the authors, license, source and modifications with the selected scene.

| Subject                 | Authors and license                                                                                                  | Prepared bytes | Preparation                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------: | ---------------------------------------------------------------------------------------------------------- |
| Toy Car                 | Guido Odendahl, Eric Chadwick; CC0 1.0                                                                               |      5,422,412 | Unmodified upstream GLB                                                                                    |
| Sheen Wood Leather Sofa | Fran Calvente (original CC0); Eric Chadwick / Darmstadt Graphics Group GmbH (improvements CC BY 4.0)                 |     10,107,912 | Unmodified upstream GLB                                                                                    |
| Fox                     | PixelMannen (original CC0); tomkranis (rig/animation CC BY 4.0); AsoboStudio and scurest (glTF conversion CC BY 4.0) |        162,852 | Unmodified upstream GLB; displayed in its static pose                                                      |
| SciFi Helmet            | Michael Pavlovic, Norbert Nopper; CC0 1.0                                                                            |      9,170,980 | Geometry/materials unchanged; textures resized to at most 1024 × 1024 and packed into a self-contained GLB |

Upstream metadata/license documents use CC BY 4.0. Attribution for that documentation: KhronosGroup, glTF Sample
Assets, linked above. The model licenses remain as listed. Damaged Helmet was excluded because the pinned source
credits earlier CC BY-NC content; the CC0 SciFi Helmet serves the helmet comparison instead.

## Reproduce and validate

Use Python 3 with `Pillow==12.3.0`, then run from the repository root:

```sh
python3 packages/website/scripts/prepare-demo-models.py
pnpm --filter website validate:models
pnpm exec vitest run --project unit packages/website
```

The preparation script downloads only pinned scene sources, verifies their SHA-256 values, and rejects any output
that does not reproduce the recorded size/hash. The helmet uses Pillow Lanczos downsampling and PNG compression
level 9; its mesh, UVs, material values and color-space interpretation are unchanged. Other GLBs are copied byte
for byte. The images are embedded; selecting a new model only requests its local GLB, with no external resources.

The official Khronos validator is pinned at `gltf-validator@2.0.0-dev.3.10`. Validation checks GLB framing, the hash
and size budget (12 MiB per new GLB), absence of resource URIs, and the reviewed extension allowlist. All four
models report zero validation errors. Toy Car has two and Sofa has six upstream tangent-space warnings: their
normal-mapped primitives lack supplied tangents and rely on the renderer's derivative-based tangent frame.
Fox and SciFi Helmet have no warnings. These warnings may cause normal-map differences across implementations.

Three.js r186 GLTFLoader supports all used extensions: texture transforms, clearcoat, transmission, sheen,
specular and WebP. Sofa requires WebP decoding; unsupported browsers display a recoverable English load error.
Fox's skin is retained; animations are not played. Embedded PBR materials are preserved; only the original head's
separate maps use the existing skin-material override.

## Framing and recovery

New subjects are centered from their actual scene bounds and scaled so the largest dimension is 2.4 scene units.
Reset camera fits their bounding sphere within both viewport axes with 12% margin. Each registry entry defines
the viewing direction and environment intensity; narrowing the viewport backs out along the current orbit if
needed. The original head keeps its existing camera, scale and lighting defaults. Scene changes reset temporal
history.

The UI retains the last working scene and attribution while another loads. Failed selections restore the working
selection, show an English message and offer retry. The separate resource-cleanup change in issue #8 provides
idempotent disposal, partial-load cleanup and same-ID request tokens; integrate it for repeated-switch ownership
and same-ID supersession guarantees. This scene PR independently rejects superseded loads of different subjects.
Unit fixtures decode actual mesh/skin/material data with image decoding stubbed, test bounds/frustum fitting and
material preservation, and test failure/retry and different-scene supersession. Pillow decoded all 26 embedded images successfully (PNG/WebP, at most 2048 × 2048). Browser
decoding/rendering remains separate validation: the Browser runtime was unavailable on this machine. No GPU
benchmark results are claimed.
