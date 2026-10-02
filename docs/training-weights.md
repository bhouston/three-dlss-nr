# Training open weights for the NR network

> **Attribution.** The network architecture is NVIDIA's: the DLSS 5 Neural Rendering (NR) network, build 310.8.0
> ([report](https://research.nvidia.com/labs/adlr/DLSS5/files/DLSS5_Report.pdf),
> [project page](https://research.nvidia.com/labs/adlr/DLSS5/)). The bit-exact open reimplementation, its numerics and
> its documentation are [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
> [maan](https://github.com/maanHimself) (MIT). This repository is a three.js (TSL / WebGPU) port of OpenDLSS-NR, pinned
> at [`9d08f41`](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41). This project is not affiliated with NVIDIA or
> with the author of OpenDLSS-NR. "DLSS" is an NVIDIA trademark.

Status: a research report for issue #1, written 2026-10-02. It does not ship code. It is for maintainers and ML or
graphics engineers who need to decide whether, and how, to train weights that this project owns, so the network can
run on something other than NVIDIA's proprietary weights.

**How to read it.** Each claim is marked as one of three kinds:

- **Known** means the claim is cited to the OpenDLSS-NR docs, this repository's code, NVIDIA's public material, or a
  paper.
- **Inference** means our reading of known facts. It is plausible but not confirmed.
- **Proposal** means something we recommend doing. Nobody has tried it on this network.

All cost figures are estimates. The assumptions behind each one are stated next to it.

## Contents

0. [Summary](#0-summary)
1. [What the network computes](#1-what-the-network-computes)
2. [Training data](#2-training-data)
3. [Objectives](#3-objectives)
4. [The temporal path](#4-the-temporal-path)
5. [A PyTorch model of the exact architecture, and FP8](#5-a-pytorch-model-of-the-exact-architecture-and-fp8)
6. [Export and validation](#6-export-and-validation)
7. [Compute and data budget](#7-compute-and-data-budget)
8. [Evaluation](#8-evaluation)
9. [Legal and ethical constraints](#9-legal-and-ethical-constraints)
10. [Phased plan](#10-phased-plan)
11. [References](#11-references)

## 0. Summary

- **Feasibility.** It is feasible to train weights for the exact architecture and export them in the format all three
  implementations load (Vulkan, the reference WebGPU port, and this TSL port). The architecture is fully specified
  down to every rounding. The model directory layout is fully specified as well. This repository already writes that
  layout for its synthetic weights ([`synthetic/generate.ts`](../packages/three-dlss-nr/src/synthetic/generate.ts)), so
  writing a trained model is mostly a matter of using different numbers.
- **The hard part is the data and the objective, not the network.** NVIDIA describes DLSS 5 as a _one-step pixel-space
  diffusion model_ trained with _appearance priors learned from real-world visual data_. It has not published its
  datasets or its losses. An open model has to build its own data and objective. Expect a first open model to be
  clearly less "photoreal" than NVIDIA's. It can still be useful and honest.
- **Network compute is cheap. Data and iteration are not.** The network has about **146 M parameters** (143.8 M FP8
  weights, 1.9 M f16 attention priors, and 47 k scales; 147,683,618 bytes = 140.8 MiB). It costs **69.4 G
  multiply-accumulates (MACs) per 512x512 frame** (about 265 k MAC per pixel, falling to about 246 k at 1080p). Under the
  assumptions in [section 7](#7-compute-and-data-budget):
  - A **minimal prototype** costs about **10-40 GPU-hours** of training per run. Rendering its data costs about
    **30-700 GPU-hours**.
  - A **credible open model** (temporal, FP8 quantization-aware, adversarial or distilled realism) costs roughly
    **0.6-3 k GPU-hours** of training including ablations. Its data costs **1-25 k GPU-hours** of path tracing.
    Distilling from an open diffusion teacher could add **1-15 k GPU-hours**.
  - In money, that is about $0.1-3 k for the prototype and $5-150 k for the credible model, at an assumed $2-4 per
    H100-class GPU-hour. Engineering time is the larger cost.
- **FP8 is a first-class training constraint.** The format has no per-tensor scale factors. Every weight is a raw
  E4M3 byte that the loader requires to satisfy |w| <= 9. Every activation that crosses a kernel boundary is a raw
  E4M3 byte as well. A model trained in float and then rounded will probably not survive. Quantization-aware training
  (QAT) that emulates the exact publication points is required.
- **Legal.** Do not train on, distil from, or evaluate against NVIDIA's model or its outputs without legal review.
  Check every dataset license, including whether ShareAlike or NonCommercial terms carry over to trained weights.
  Avoid real people's likenesses unless they have consented.
- **Recommended first step.** [Phase 0](#10-phased-plan):
  1. Write the PyTorch model, plus an importer and exporter for the model directory.
  2. Prove the PyTorch model against **this repository's synthetic weights**. They have golden outputs that were
     verified byte-for-byte against the reference, so the check needs no NVIDIA data.
  3. Overfit a few hundred synthetic image pairs, export the result, and run it in the TSL network and the demo.

  This takes a few person-weeks and a handful of GPU-hours. It retires the format and numerics risk before anyone
  spends money on data.

## 1. What the network computes

### 1.1 Inputs, outputs and composition (Known)

From [network.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/network.md) and
[frame.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/frame.md):

| lanes | input (f32 per padded pixel)                                                                                                                                                                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0-2   | Three Gaussian noise lanes. They come from a hash of the padded pixel coordinate and a per-frame seed. The reference uses `frameIndex & 0xffff`, so the noise changes every frame.                                             |
| 3     | The constant 1.                                                                                                                                                                                                                |
| 4-6   | The display proxy, centred: `f16((f16(c) - 0.5) * 0.125)`. `c` is the sRGB code value of `scene / paperWhite`, with a soft shoulder above 0.75.                                                                                |
| 7-9   | The reprojected previous **output**, transformed the same way. When there is no history, these lanes copy lanes 4-6.                                                                                                           |
| 10    | `style / 128`. Style is an id (the demo uses 0 none, 1 cinematic, 2 natural).                                                                                                                                                  |
| 11    | Local tone.                                                                                                                                                                                                                    |
| 12-14 | The structure / skin / auto-mask triple. With auto-mask off it is `(localStructure, -1, -1)`. With auto-mask on it is `(1, skinStructure or localStructure, localStructure)` (`preprocess.wgsl` lines 80-85 in the reference). |
| 15    | 0.                                                                                                                                                                                                                             |

The output is 4 f32 values per pixel from a `32 -> 4` f16 head. The frame pipeline then does this:

```
neural   = clamp(proxy + rgb / 4, 0, 1)                 # sRGB code space
weight   = clamp(sigmoid(logit) * blendScale, 0, 1)     # blendScale = block70.layer0.blend_scale (learned f16)
display  = lerp(neural, reprojected_history, weight)    # only where a history exists
history' = truncate_to_half(display)
```

Key properties:

- **Same resolution in and out.** NR is not an upscaler. It re-renders a tone-mapped LDR proxy of the frame. Outside the
  valid rectangle the image is mirrored, while the noise keeps using the padded coordinate.
- **Noise is an input; the network adds detail, it does not remove noise.** In OpenDLSS-NR's words, it _"re-renders a
  tone-mapped proxy of the frame, generating detail from the three injected Gaussian lanes under the tone, structure,
  skin and style conditioning; it is given noise and does not remove any."_
- **No G-buffer inputs.** The network does not see depth, normals, albedo or material ids. Motion vectors are used only
  outside the network, to reproject the history. Every conditioning value is a **per-frame scalar**, not a per-pixel
  mask. Anything mask-like, such as "skin", has to be recognised by the network itself.
- **The temporal path is a learned per-pixel blend.** The network decides, per pixel, how much reprojected history to
  keep. A global learned cap (`blendScale`) limits it. NVIDIA's shipped value is 0.7397 (frame.md).

### 1.2 Architecture (Known)

The network is a U-net of shifted-window transformer blocks with a global ViT at the bottom. It has 71 blocks over
six pooling levels.

| blocks        | level     | channels | notes                                                                             |
| ------------- | --------- | -------- | --------------------------------------------------------------------------------- |
| 0, 70         | field     | 32       | block 0 has the `16 -> 32` f16 input adapter; block 70 has the `32 -> 4` f16 head |
| 1-4 / 66-69   | L0 (1/2)  | 32       | dense FFN `32 -> 128 -> 32`                                                       |
| 5-8 / 62-65   | L1 (1/4)  | 64       | grouped FFN: C/32 paths `C -> 128 -> 32`, concatenated, then `C -> C`             |
| 9-14 / 56-61  | L2 (1/8)  | 128      | same                                                                              |
| 15-22 / 48-55 | L3 (1/16) | 256      | same                                                                              |
| 23-30 / 40-47 | L4 (1/32) | 512      | 8 branches `64 -> 256 -> 64`, then `512 -> 512`                                   |
| 31-38         | L5 (1/64) | 1024     | ViT: global attention over all tokens; FFN `1024 -> 4096 -> 1024`; 32 heads       |
| 39            | L5 -> L4  | -        | the `1024 -> 512` projection out of the ViT                                       |

The block-level facts that matter for training:

- **Block structure.** A block is `y = x * ffnScale + FFN(x)` followed by
  `out = y * attnScale + Proj(Attn(QKV(y)))`. `ffnScale` and `attnScale` are learned per-channel f16 vectors.
- **No normalisation layers.** There is no LayerNorm or RMSNorm anywhere. The only normalisation is the cosine
  normalisation of q and k inside attention.
- **Scaled cosine attention.** q and k are L2-normalised, then multiplied by a learned per-head f32 temperature. The
  window blocks also add a learned 64x64 per-head prior (bias). This is the scaled cosine attention of Swin
  Transformer V2 [Liu 2021b], but with a free 64x64 bias instead of Swin's log-spaced relative-position MLP.
- **Windows.** Windows are 8x8. They shift through four phases, (0,0), (-4,-4), (-4,0), (0,-4), and the phase continues
  from an encoder stage into its matching decoder stage. Out-of-field tokens are zero vectors that still enter the
  softmax denominator.
- **The softmax exponential is an approximation.** It is a bit trick equal to `2^(1.4375 s - 5.375)`, which is about
  `exp(0.996 s)`, with the score clamped to roughly [-6, 6]. The ViT uses `2^(1.4326 s - 3.6502)` with a clamp of
  roughly [-3, 3]. There is no max subtraction.
- **The activation is a cubic polynomial, not SiLU.** The reference calls it `mpCubicSilu`:
  `x * (0.89453125 + b * (0.447265625 - 0.055908203125 * |b|))`, where `b = clamp(x, -4, 4)`
  ([numerics.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/numerics.md),
  `ports/browser-webgpu/src/numerics.js`).
- **Transitions.**
  - Encoder: a 2x2 average pool, then a `C -> 2C` GEMM.
  - Decoder: a `2C -> C` GEMM, a nearest 2x upsample, then `+ skip * transitionScale`.
  - Final merge: `up * inputScale + block0 * adapterScale`.

**Inference.** The "Mp" in the reference's name and the overall design suggest a magnitude-preserving design in the
style of EDM2 [Karras 2023]: no normalisation layers, learned scales on the residual stream, cosine attention, and a
SiLU-like activation of magnitude-preserving size. For comparison:

| x      | `mpCubicSilu(x)` | EDM2 `silu(x) / 0.596` |
| ------ | ---------------- | ---------------------- |
| near 0 | about 0.89 x     | about 0.84 x           |
| 4      | 7.15             | 6.59                   |

This is a hint about NVIDIA's training recipe, not evidence. It is still a sensible place to start a recipe for an
open model (section 5.2).

### 1.3 Numerics that training must respect (Known)

From [numerics.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/numerics.md) and
[weights.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/weights.md):

- **Activations.** Every value that crosses a kernel boundary is published as **E4M3** (max 448; normal range starts
  at 2^-6, subnormals go down to 2^-9). It is rounded to half first, then to E4M3, round-to-nearest-even at both
  steps, saturating. Intermediates and the 32-channel residual stream are **f16**.
- **Weights.** Every GEMM weight is a **raw E4M3 byte**. There is no per-tensor or per-channel weight scale in the
  format. The loader rejects any weight with |w| > 9, which is magnitude code > 0x51
  ([`Model.ts` `firstUnboundedWeight`](../packages/three-dlss-nr/src/model/Model.ts)). The bounded-half GEMM is exact
  only under that bound.
- **Accumulation.** GEMMs accumulate in f16 with the hardware's F13 fixed-point grouping, in a fixed K order. The
  residual is the accumulator's seed.
- **Other parameters.** Skip and transition scales are f16. Attention temperatures are f32. Priors are f16. The two
  small matrices (adapter and head) are f16.

**Consequence (Inference).** The trained real-valued function is _not_ the shipped model. The shipped model is the
sequence of these roundings. A training model has to emulate at least the E4M3 publications, the f16 head, the
exponential approximation and its clamps, and the polynomial activation. Otherwise the exported weights will behave
differently from the trained ones. Section 5 covers how.

## 2. Training data

### 2.1 What NVIDIA says (Known, and very little)

The DLSS 5 report PDF returned **HTTP 403** to automated fetches on 2026-10-02, so this section relies on what is
publicly readable. That is the project page abstract, NVIDIA's developer blog post, and press coverage. The **report
itself should be read by a human before Phase 2.** It may contain details that would change the recommendations below.

The project page says:

- DLSS 5 _"complements conventional rendering with appearance priors learned from real-world visual data"_.
- It uses _"a one-step pixel-space diffusion model"_.
- _"At inference, the model is conditioned on the current rendered frame, engine motion vectors, carried temporal
  state, and artistic-direction values."_
- _"During training, consistency supervision from renderer-derived scene attributes further grounds generation to
  remain faithful to the authored scene."_
- It is _"causal and deterministic"_ and _"trained for frame-to-frame temporal stability"_.

NVIDIA's developer blog adds the controls:

- _"Structure Intensity and Tone Intensity to tune high-frequency detail and broader lighting and color response"_.
- _"semantic AI masking to apply or hold back the effect across recognized scene elements"_.
- _"choose from among several models"_.

IEEE Spectrum reports that _"Nvidia has so far kept a lid on details of the underlying training datasets"_.

What that tells us:

- **(Inference)** The objective is generative. Paired supervision alone does not produce a one-step diffusion model.
  The noise lanes are its latent input.
- **(Inference)** The targets are anchored in real photographs, not only in renders.
- **(Inference)** Renderer data (G-buffers, segmentation) is used only in losses, which matches the network having no
  G-buffer inputs.
- **(Unknown)** The dataset scale, the dataset sources, the loss weights, and whether a large teacher was distilled.

### 2.2 The pair the network needs

One training sample needs:

- **Input:** an LDR proxy of a rendered frame (lanes 4-6), built exactly as the runtime builds it. That means
  `scene / paperWhite`, the 0.75 shoulder, sRGB encoding and f16 rounding (frame.md).
- **Conditioning scalars:** lanes 10-14.
- **For temporal training:** the reprojected previous output and a history-valid mask.
- **A target or a critic:** what the "enhanced" frame should look like.

In addition, losses can use renderer attributes that the network itself never sees: motion, depth, normals, albedo,
material or segmentation ids, and a skin mask.

There are three realistic open ways to provide the target.

#### Option A: synthetic paired data, rasterised to path-traced (Proposal; recommended first)

Render the same scene, camera and animation twice:

- **Input:** a real-time rasteriser close to what users of this library run. That is three.js `WebGPURenderer`
  (MeshStandardMaterial / MeshPhysicalMaterial, shadow maps, environment lighting). EEVEE is an alternative.
- **Target:** an offline path tracer at high samples per pixel. Blender Cycles is the obvious choice, or
  three-gpu-pathtracer to stay inside three.js. Use the same assets, preferably with richer materials (subsurface
  scattering, hair, thin-film) and real global illumination.

Pros:

- The input and target align exactly per pixel, and the renderer gives free G-buffers, motion vectors, skin masks and
  sequences.
- Licensing is controllable: use only CC0 or CC BY assets.
- The domain gap to deployment is small, because the input is literally a three.js frame.

Cons:

- The realism ceiling is the path tracer's and the assets', not reality's. The model learns "make this raster frame
  look path-traced", which is a real and useful effect: GI, soft shadows, SSS, specular occlusion. It is not "make it
  look like a photo".
- Asset diversity is the bottleneck.

Asset sources:

- [Poly Haven](https://polyhaven.com/license): CC0 models, textures and HDRIs.
- BlendKit CC0 assets.
- [Smithsonian Open Access](https://www.si.edu/openaccess): CC0 3D scans.
- Sketchfab CC BY models.
- The heads in [`docs/suggested-assets.md`](suggested-assets.md). There are only a handful, so they are better used as
  **evaluation** scenes than as training data.

Apply the same license policy as that file: no NC, no ND, and watch for "NoAI" clauses.

The existing Hypersim dataset [Roberts 2021] is a large ready-made source of path-traced HDR frames:

- 77,400 images of 461 indoor scenes. The public release has 74,619 images, after removing images with people and
  logos.
- It includes diffuse reflectance, diffuse illumination and a non-diffuse residual per image, from which a cheaper,
  raster-like input could be built.
- It is licensed **CC BY-SA 3.0**. Whether ShareAlike reaches trained weights needs legal review (section 9).
- It has single frames only, with no sequences.

#### Option B: unpaired realism from photographs (Proposal; Phase 2+)

Keep the rendered input. Replace the per-pixel target with a critic trained on real photographs, plus a
content-preservation term. This is the approach of Intel's "Enhancing Photorealism Enhancement" [Richter 2021]. It
translated GTA V frames toward Cityscapes, KITTI and Mapillary Vistas. It used a discriminator conditioned on
semantic labels, and the generator used G-buffers. Other unpaired methods include CycleGAN [Zhu 2017] and CUT
[Park 2020].

This option comes closest to NVIDIA's "priors learned from real-world visual data". It is also the hardest to make
stable and temporally consistent, and to keep from hallucinating.

Photo licensing is the main constraint:

- Cityscapes and Mapillary Vistas carry non-commercial terms.
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)'s README says its licenses allow free use _"for non-commercial
  purposes"_, and some of its images are CC BY-NC 2.0.

An open model that is meant to be redistributable needs photo sets with permissive terms, such as CC0 or CC BY
collections that are filtered for license. Someone has to verify the license of each one.

#### Option C: distilling a diffusion prior (Proposal; research-grade)

Use an openly licensed image diffusion model [Rombach 2022] as a realism teacher, conditioned on the render through
something like ControlNet [Zhang 2023] or SDEdit-style partial noising. Then train the 146 M-parameter network as a
one-step student, using adversarial diffusion distillation [Sauer 2023], distribution matching distillation
[Yin 2023], or the one-step image-to-image recipe of pix2pix-turbo [Parmar 2024].

This matches NVIDIA's description ("one-step diffusion") best. It also carries the most risk:

- The teacher's own license and training data become part of our model's provenance.
- Teacher compute dominates the cost (section 7).
- Holding temporal stability under a generative prior is an open research problem for us.

### 2.3 Conditioning data (Proposal)

The meaning of NVIDIA's scalars is known only from their names and NVIDIA's blog. An open model can _define_ them. It
should keep the defaults that the pipeline already sends, so existing integrations keep working.

| scalar                      | proposed semantics for an open model                                                                                                 | training signal                                                                                                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| localTone (lane 11)         | Strength of the low-frequency change (lighting and colour response). 0 means none; 1 (the default) means full.                       | Sample t from [0, 1.5]. Build the target's low-pass band as `lerp(input_lp, target_lp, t)`.                                                                                                                                  |
| localStructure (lane 12/14) | Strength of the high-frequency change (detail and micro-shadowing). 0 means none; 1 (the default) means full.                        | Sample s. Build the target's high-pass band as `lerp(input_hp, target_hp, s)`.                                                                                                                                               |
| skinStructure (lane 13)     | With auto-mask on, a separate structure strength for skin.                                                                           | A per-pixel target built from a **skin mask that the renderer supplies at training time only**. The network has to learn to find skin itself. Synthetic humans give exact masks.                                             |
| style (lane 10)             | Either leave it unused (always 0) and keep the post-network grade that the composite already applies, or train a few discrete looks. | Discrete target variants. Note that style / 128 is a very small input (0.0078 per step), so the network has to learn to separate those values. Mapping new looks to new ids is cheap only if the training data defines them. |

Sample the scalars per training example, and include 0 so that "off" is learned as near-identity. That gives users a
real control. Without this, a trained model would ignore the lanes, and the controls in the demo UI would do nothing.

## 3. Objectives

These are all proposals. They are ordered from simplest to most ambitious. Train the **head output through the
composition**, `clamp(proxy + rgb / 4, 0, 1)`, rather than the raw residual, so that the loss sees what the user sees.

1. **Reconstruction (Option A).**
   - Use L1 or Charbonnier in proxy code space, plus a loss in linear light to stop highlights from washing out.
   - Clamping kills gradients. Use a soft clamp in training, or a small penalty on how far the residual goes past the
     clamp.
2. **Perceptual.**
   - LPIPS [Zhang 2018] or DISTS [Ding 2020] against the target. These stop pure-L1 training from blurring the result.
   - Use them with care: a perceptual metric that is also a training loss is no longer a fair metric (section 8).
3. **Adversarial.**
   - A patch discriminator on the composed output against targets (Option A) or photos (Option B). A small weight on
     top of 1 and 2 is the classic recipe that SRGAN [Ledig 2017] introduced.
   - For B, condition the discriminator on semantic labels, as in [Richter 2021]. That prevents "trees become buildings"
     style hallucination.
4. **Content and structure consistency, using renderer attributes ("consistency supervision" in NVIDIA's words).**
   - Penalise changes that the scene does not justify: edges that are absent from the renderer's depth, normal and id
     edges, or semantic labels that a frozen segmenter reads differently from the input's ids.
   - For faces, an identity-embedding distance between input and output, evaluated only on consented or synthetic
     identities.
5. **Noise-driven detail.**
   - Lanes 0-2 are the latent. In a one-step diffusion or distillation setup (Option C), the noise is what the student
     maps to a sample.
   - In Options A and B, a pure regression loss teaches the network to **ignore** the noise, because the conditional
     mean is noise-free. A noise-conditioned adversarial loss or a diversity term is needed if the noise should mean
     anything.
   - Since the noise changes every frame (section 1.1), any detail the network "generates" from noise flickers unless
     the temporal path anchors it. A simpler open model may reasonably learn to ignore the noise. That is acceptable
     for v1.
6. **Conditioning control.** The scalar-sampled targets of section 2.3, plus an identity loss at scalar 0.
7. **Temporal.** See section 4.
8. **Quantization.** An activation-range regulariser and the QAT terms in section 5.3.

## 4. The temporal path

**Known.** The network sees the reprojected previous output (lanes 7-9) and emits a blend logit. The composite blends
with weight `clamp(sigmoid(logit) * blendScale, 0, 1)`, and only where a history exists. The history is stored
truncated to half. It is reprojected with motion vectors and a 5-tap Catmull-Rom filter (frame.md). Disocclusions keep
a valid motion vector, so the history at those pixels shows the wrong surface. **The network's logit is the only thing
that rejects it** (frame.md: on Bistro, NVIDIA's model's mean blend weight falls from 0.69 at rest to 0.15 during an
orbit).

**Proposal.** Train in recurrent sequences that reproduce the runtime exactly.

1. **Data.** Use short clips of 4-16 frames with exact per-pixel motion vectors, a history-valid flag (off-screen
   previous position means invalid), and, for the loss only, a disocclusion mask from the renderer's previous-frame
   depth or id buffer. Mix camera motion, object motion, skinned animation and static shots. Static shots matter: with
   changing noise and a still camera, any flicker is the network's fault.
2. **Unrolling.** Feed the model's **own** previous composed output back as history, through the same warp and
   Catmull-Rom filter and the same f16 truncation. Feeding the target as history instead (teacher forcing) creates
   exposure bias: the model never learns to correct its own drift.
   - Use truncated back-propagation through time over K = 2-4 frames, after a no-gradient warm-up of a few frames so
     the history is "lived in".
   - Randomly reset the history (lanes 7-9 = proxy, weight 0) on about 10% of frames, because the runtime does that
     after cuts and resizes.
3. **Losses.**
   - The per-frame losses of section 3 on the composed `display`.
   - A **temporal warping error** between `display_t` and `warp(display_{t-1})` on pixels that are not occluded, in
     the style of Lai et al. [Lai 2018]. Weight it relative to the target's own temporal change, so that real lighting
     changes are not punished.
   - A ping-pong or recurrent consistency term in the style of TecoGAN [Chu 2018] to stop slow drift.
   - Supervision for the blend logit, implicit through the losses above, plus a weak explicit term: push the weight
     toward 0 on known disocclusions and toward `blendScale` on static, visible pixels.
4. **The noise schedule.** Use a fresh noise seed every frame, as the runtime does. Holding the seed fixed in training
   would hide flicker that users will see.

## 5. A PyTorch model of the exact architecture, and FP8

### 5.1 Building it from the specification (Proposal, with Known constraints)

Write a `torch.nn.Module` with one submodule per manifest record, so that export is a 1:1 walk. The structure comes
from [network.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/network.md). The exact parameter shapes
and byte offsets come from [`layouts.ts`](../packages/three-dlss-nr/src/model/layouts.ts) (`modelRecords()`: 153
records, each with typed regions).

| piece                     | what to implement                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| geometry                  | Port `geometryFromValid` ([`geometry.ts`](../packages/three-dlss-nr/src/geometry.ts)): the padded field, six levels, each rounded up to a multiple of 4, mirrored padding. Training on crops whose field equals the crop avoids padding waste. Still train on some padded sizes, because inference always pads.                                                                                                                |
| window attention          | Partition into 8x8 windows with the four-phase origin counter (`takeWindowPhase`). Out-of-field tokens are **zero vectors that stay in the softmax** (no masking). q and k are cosine-normalised, then multiplied by the per-head temperature. Add the 64x64 prior. Use the exponential `2^(1.4375 s - 5.375)` with s clamped to [-6, 6]. The ViT uses its own constants, the sqrt(32) factor, and the padding-row correction. |
| FFN variants              | Dense (32), grouped bottleneck (64-256), 8-branch (512), dense 4096 (ViT). The activation is `mpCubicSilu`, not `torch.nn.SiLU`.                                                                                                                                                                                                                                                                                               |
| residual                  | `x * ffnScale + FFN(x)` then `y * attnScale + attn(y)`. Note the asymmetry: the 32-channel blocks add the raw f16 FFN result to the attention skip, while the wider blocks add its E4M3 publication.                                                                                                                                                                                                                           |
| transitions and merges    | The exact pool, GEMM and upsample order of network.md "Transitions".                                                                                                                                                                                                                                                                                                                                                           |
| head and composite        | The `32 -> 4` head, then `clamp(proxy + rgb/4)`, `sigmoid * blendScale` and the history lerp, all in the training graph.                                                                                                                                                                                                                                                                                                       |
| initialisation (no norms) | Start the skip scales near 1 and initialise each branch's last layer near zero, in the style of ReZero / Fixup / LayerScale [Bachlechner 2020; Zhang 2019; Touvron 2021]. Then the 71-block stack starts close to identity and is stable without normalisation.                                                                                                                                                                |
| attention prior           | Parameterise it as a relative-position table (Swin) or a small MLP, and **expand it to the full 64x64 matrix at export**. The format stores the full matrix, so any parameterisation that expands to it is allowed.                                                                                                                                                                                                            |

Keep f32 master weights and optimizer state, and compute in bf16 mixed precision. Compile the model, or use FlexAttention or
custom kernels for the windowed attention. The 32-channel full-resolution levels are memory-bound, so expect low
utilisation there (section 7).

### 5.2 Verifying the PyTorch model without NVIDIA weights (Proposal; the key Phase 0 test)

This repository already has a ground truth that needs no NVIDIA data:

- Deterministic synthetic weights ([`generateSyntheticModel`](../packages/three-dlss-nr/src/synthetic/generate.ts),
  `node scripts/make-synthetic-model.mjs <dir>`).
- Deterministic synthetic input features ([`syntheticFeatures`](../packages/three-dlss-nr/src/synthetic/features.ts)).
- Golden digests of every block boundary and of the head, verified byte-for-byte between the reference WebGPU port and
  the TSL port in Chrome (`test/browser/run-network-parity-chrome.mjs`, `test/network/goldenDigests.ts`).

The test:

1. Write an **importer**, the inverse of the exporter in section 6, that loads the synthetic model directory into the
   PyTorch module.
2. Run the module in a **numerics-emulation mode** on the synthetic features. In that mode every E4M3 and f16
   publication point of numerics.md is a fake-quant op, and the exponential, polynomial and clamps are exact.
3. Compare every boundary and the head with the runtime. Use boundary dumps from `NRNetwork` with
   `captureBoundaries: true`.

Bit-exactness is **not** expected. Faithful emulation of the F13 truncated accumulation and the fixed K order is
possible but slow. Instead, set a tolerance:

- Head PSNR above about 45 dB.
- Most E4M3 boundary bytes within one code.

Then track how that gap grows over the 71 blocks. This one test proves the importer, the exporter's inverse, the block
wiring, the window phases and the prior layout, all before any training.

### 5.3 FP8 E4M3 quantization-aware training (Proposal)

**Why it is needed (Inference from Known facts).** There are no scale factors in the format, so magnitudes have to fit
E4M3's grid natively. The arithmetic for weights:

- With a He-style weight standard deviation of `1/sqrt(K)`, K = 4096 (the ViT contract) gives 0.0156, which is exactly
  E4M3's smallest normal (2^-6).
- So about half of those weights would be subnormal, with 2 or fewer mantissa bits.
- About 5% would flush to zero, because |w| < 2^-10 means |z| < 0.0625 for a standard normal z.
- K = 512 gives 0.044, which is comfortable. K = 32 gives 0.18.

The synthetic generator notes the same effect: "E4 subnormals occur naturally for large K"
([design notes, §5.1](#11-references)). For activations, OpenDLSS-NR's synthetic calibration found that the residual
stream either collapses into subnormals or saturates unless the skip scales sit in a narrow band
([`gains.ts`](../packages/three-dlss-nr/src/synthetic/gains.ts)). A trained model faces the same cliff.

**Recipe.**

1. **Fake-quant at every publication point** in the forward pass: f32 -> f16 -> E4M3 with round-to-nearest-even and
   saturation at 448, using a straight-through estimator (STE) for gradients.
   - Weights: fake-quant to E4M3 with **|w| clamped to 9**.
   - Skip scales, priors and the head: fake-quant to f16. Attention temperatures stay f32.
   - Build the E4M3 codes as `x.to(float16).to(float8_e4m3fn)`, and **verify the cast** against this repository's
     `e4m3FromNumber` oracle over all 65,536 halves. PyTorch's cast may not saturate the way the runtime does, so
     clamp first. Also check the NaN and signed-zero behaviour.
2. **Learn magnitudes inside the grid.**
   - Use reparameterisations that are exact at export: weight = learnable per-tensor scale times a normalised
     direction (EDM2's forced weight normalisation [Karras 2023]). **Fold the scale into the stored E4M3 values** before
     export, since the format has nowhere else to put it.
   - Only the f16 skip, transition and adapter scales, the attention temperatures and the head can absorb scale for
     free. Cosine attention is scale-invariant in q and k. The FFN's first layer is not, because of the activation.
3. **An activation-range regulariser.** Penalise each boundary's RMS leaving roughly [2^-4, 8] and its saturated
   fraction going above 0.1%. These are the same thresholds as the repository's calibration gate
   ([`syntheticStats.gpu.test.ts`](../packages/three-dlss-nr/src/synthetic/syntheticStats.gpu.test.ts)).
4. **Schedule.** Train in bf16 or f32 first. Switch QAT on for the last 10-30% of steps, then end with a short
   low-learning-rate phase at full emulation (exponential, polynomial, half residual). Learned step-size quantization
   [Esser 2019] is not directly applicable, because there is no step to learn. The scale-folding in step 2 plays that
   role.
5. **Post-training calibration** (no QAT) is worth trying as a baseline: rescale each layer to centre its weight
   histogram in E4M3's normal range, and push the compensation into the f16 scales. **Expect it to lose noticeably to
   QAT**, given the no-scale format.

On the FP8 literature: [Micikevicius 2022] defines E4M3 and E5M2 and shows FP8 training and inference matching
higher-precision results _with per-tensor scaling_. NVIDIA's [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
and DeepSeek-V3 [DeepSeek-AI 2024] train with fine-grained FP8 scaling. This format's lack of scales is unusual, which
is the main reason QAT is not optional here.

Training itself does not need FP8 hardware. Fake-quant in bf16 is enough. Hopper or Blackwell FP8 GEMMs could speed
up training, but would complicate emulation.

## 6. Export and validation

### 6.1 Writing the model directory (Known format, Proposal tooling)

The format is the one [`synthetic/generate.ts`](../packages/three-dlss-nr/src/synthetic/generate.ts) already writes.
An exporter in Python, or a small TypeScript tool that takes a `.safetensors` file of logical tensors, should mirror it.

**Directory layout.** `manifest.json` plus `model/stages/s00.bin ... s10.bin`. The manifest has:

- `totals.blockCount: 71`.
- 11 `stages`, each with `id`, `file`, `packedByteLength` and `sha256`.
- 153 `tensors`, each with `name`, `block`, `layer`, `stage`, `stageOffset` and `byteLength`.

`format` is free text, for example `"three-dlss-nr open v0"`. Record offsets are 16-byte aligned. Bytes outside every
region are zero.

**Record lengths.** Take them from `modelRecords()` / `recordLayout(name)`. The lengths of block 0, of the first
decoder blocks (48, 56, 62, 66) and of block 70 are checked **exactly**. Other records must be at least `readEnd`
bytes. `block70.layer0.blend_scale` is a separate 2-byte record.

**FP8 matrices.** A logical matrix `W[j][n]` has input channel j in natural order and output n. Every logical weight
goes to the byte below. The chained-index permutation stays inside 32-groups, so the same formula covers batched
matrices with the global j.

```python
# packed_weight_index / inverse_packed_input_index are ports of oracle.ts (packedWeightIndex,
# inversePackedInputIndex); this is the inverse of OpenDLSS-NR nr_model.cpp Model::fp8MatrixBytes.
codes = e4m3_codes(clamp(W, -9, 9))        # via f16, RNE, never 0x7f/0xff
for j in range(K):
    for n in range(N):
        out[offset + packed_weight_index(inverse_packed_input_index(j), n, N)] = codes[j, n]
```

**Other regions.**

- **f16 matrices** (the adapter, `16 -> 32`, and the head, `32 -> 4`): `packedF16WeightIndex(k, n, N)`, in whole
  16x16 tiles.
- **Attention prior:** the exact inverse of `relayoutRelativeBias` in [`Model.ts`](../packages/three-dlss-nr/src/model/Model.ts).
  Both token axes are in the 4x4-tiled order (`tiledToken`), stored as 16x16 fragments.
- **Window temperatures:** f32 per head, in an `alignUp(4 * heads, 16)` slot.
- **ViT temperatures:** f32 per head, at the **start** of `layer2`.
- **Skip scales:** f16 per channel. `blend_scale` is a single f16.
- **The ViT's `layer3`:** two unread bytes. Write any finite half.

**Unconfirmed: the QKV column order.** docs/weights.md says only "fused Q|K|V projection, head-major". The exact
column order has to be taken from the reference's attention kernels, and it must be proven by the Phase 0 round trip.
That round trip is the guard against every packing mistake, not only this one.

### 6.2 Validation with this repository's tools (Known tools, Proposal use)

| check               | tool                                                                                                                                                                                                                                              | gate                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| format              | `NRModel.load(dirOrFiles, { verify: true })`. It runs `validateManifest` (block count, 153 records, exact or minimum lengths, stage bounds) and checks the stage SHA-256s.                                                                        | no problems                                                               |
| weight bound        | `firstUnboundedWeight`, applied as each FP8 matrix is created ([`Model.ts`](../packages/three-dlss-nr/src/model/Model.ts)). Create a network once to exercise every matrix.                                                                       | no weight > 9, no NaN code                                                |
| round trip          | Export, then import, then compare to the PyTorch tensors (section 5.2).                                                                                                                                                                           | identical codes                                                           |
| numerics gap        | Compare the PyTorch emulation's head and boundaries against `NRNetwork` on the same features (section 5.2).                                                                                                                                       | PSNR threshold, set in Phase 0                                            |
| boundary health     | `run-network-parity-chrome.mjs --model <dir> --stats-only` prints the reference's statistics for each boundary against the calibration thresholds (saturated < 0.1%, zeros < 30%, median in [2^-4, 8], 40 or more distinct codes).                | a health report, not a hard gate: a trained model may legitimately differ |
| runtime parity      | `run-network-parity-chrome.mjs --model <dir> --sizes 64x64,512x512`: the reference WebGPU port and the TSL port, byte for byte, on **our** weights.                                                                                               | bit-exact (it is today on synthetic weights)                              |
| regression fixtures | Record fixtures with the reference port on our model and run `network.real.gpu.test.ts` with `NR_WEIGHTS` / `NR_FIXTURES`. **An open model makes the "real weights" tests runnable in CI for the first time**, if the 141 MiB download is cached. | bit-exact                                                                 |
| end to end          | The website demo with the model directory, NR on and off, on the suggested heads and test scenes.                                                                                                                                                 | visual review and the metrics of section 8                                |

The Vulkan implementation can run the same directory too (`dlss5vk bench --model <dir>`). That gives a third,
independent check, on NVIDIA Ada hardware.

## 7. Compute and data budget

### 7.1 Model size (computed)

The numbers come from `modelRecords()` and its regions (a script over `dist/model/layouts.js`):

| item                                                                | count                                             |
| ------------------------------------------------------------------- | ------------------------------------------------- |
| record bytes (153 records)                                          | 147,683,618 B = 140.8 MiB                         |
| FP8 weights                                                         | 143,831,040                                       |
| of which: expand / contract / qkv / projection / merge / transition | 45.0 M / 36.5 M / 41.6 M / 13.9 M / 5.5 M / 1.4 M |
| f16 attention priors (heads x 64 x 64)                              | 1,875,968                                         |
| f16 / f32 scales and temperatures                                   | 47,466                                            |
| f16 adapter and head                                                | 640                                               |
| **total parameters**                                                | **about 145.8 M**                                 |

The brief's "about 148 M" treats every byte as a parameter. Since the priors take 2 bytes each, the true count is
about 145.8 M.

### 7.2 MACs per frame (computed)

How the count was made:

- Every GEMM region costs `K x N` MACs per token at the level where it runs. Encoder transitions run after the pool,
  and decoder upsample GEMMs run at the lower level.
- Window attention costs `2 x 64 x C` MACs per token (QK^T and PV over 64 keys).
- The ViT costs `2 x paddedTokens x 1024` MACs per token.
- Elementwise work is excluded.

| valid size | field     | ViT tokens | GMAC / frame | of which attention | MAC / valid pixel     |
| ---------- | --------- | ---------- | ------------ | ------------------ | --------------------- |
| 256x256    | 320x320   | 64         | 28.2         | 2.8                | 431 k (padding waste) |
| 512x512    | 576x512   | 96         | 69.4         | 8.1                | 265 k                 |
| 768x768    | 832x768   | 192        | 148.3        | 17.7               | 251 k                 |
| 1920x1080  | 1920x1152 | 640        | 510.4        | 65.6               | 246 k                 |

Sanity check: 2 x 510 GMAC in 7.8 ms is about 131 TFLOPS of FP8. That is plausible for the RTX 4070 SUPER figure in
OpenDLSS-NR's README. The work is spread evenly across levels: about 10 GMAC per level at 512x512.

One training issue follows from the ViT's global attention. It sees 96 tokens at 512x512 but 640 at 1080p and 2,160 at
4K. Training only on small crops would leave it untested at deployment token counts. **Include some large-crop or
full-frame steps** (Proposal).

### 7.3 Training compute

Assumptions:

- One training frame-sample costs 3x the forward pass (forward and backward): 3 x 2 x 69.4 GMAC, about 0.42 TFLOP at
  512x512.
- LPIPS (VGG at 512x512), a patch discriminator and QAT fake-quant overhead add about 0.4 TFLOP more. **Total: about
  0.8 TFLOP per 512x512 frame-sample.**
- An H100-class GPU sustains **100-250 TFLOPS effective** on this model. The 32-channel full-resolution layers and the
  window gather are memory-bound, so utilisation will be low. **Measure this in Phase 0.**
- The price is **$2-4 per GPU-hour** (an assumption about cloud prices, not a quote).

| scenario                                                                                                                                                                     | frame-samples | FLOP   | GPU-hours / run                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------ | ---------------------------------------- |
| minimal prototype: 512x512, single frame, L1 + LPIPS, 100 k steps x batch 16                                                                                                 | 1.6 M         | 1.3e18 | **1.4-3.6** (about 10-40 for a few runs) |
| credible: 1 M steps x batch 32 x 4-frame unroll, adversarial, QAT, plus about 20% for multi-resolution fine-tuning                                                           | 128 M (+20%)  | 1.2e20 | **130-340**                              |
| credible, with ablations and sweeps (x5-10)                                                                                                                                  |               |        | **0.6-3 k**                              |
| Option C teacher (distillation): an openly licensed diffusion teacher at 1-10 B parameters, with teacher and auxiliary passes **assumed** to cost 5-50x the student per step | as above      |        | **+1-15 k**                              |

The arithmetic for the minimal row: 1.6e6 x 0.8e12 = 1.28e18 FLOP. Divided by 2.5e14 FLOP/s that is 5.1e3 s, about
1.4 h. Divided by 1.0e14 it is 1.28e4 s, about 3.6 h.

The arithmetic for the credible row: 1.28e8 x 0.8e12 x 1.2 = 1.23e20 FLOP. Divided by 2.5e14 that is 4.9e5 s, about
137 h. Divided by 1.0e14 it is 1.23e6 s, about 341 h.

The network is small next to modern generative models, and **one training run is not the expensive part.**

### 7.4 Data budget (Option A)

Assumptions:

- Cycles path-traces one 512x512 frame with denoising at 1-4 k samples per pixel in **5-60 s** on one GPU, depending
  on the scene. Hair, SSS and caustics push it to the high end.
- The raster input is negligible by comparison.

| scenario  | clips x frames            | frames    | render GPU-hours |
| --------- | ------------------------- | --------- | ---------------- |
| prototype | 2.5-5 k clips x 8 frames  | 20-40 k   | **30-670**       |
| credible  | 50-200 k clips x 8 frames | 0.4-1.6 M | **0.6-27 k**     |

Rendering at 1024x1024 and cropping gives more training views per render. Storage is about 3-6 MB per frame for an
EXR target, an 8-bit input and G-buffers, so the credible set is **roughly 1-10 TB**.

The real cost is **assets**. A few thousand diverse scenes, with humans, foliage, materials, lighting and motion, is
person-months of curation and license tracking. That probably outweighs all compute.

### 7.5 Totals (rough)

| phase                                       | GPU-hours | at $2-4/h | people                                                  |
| ------------------------------------------- | --------- | --------- | ------------------------------------------------------- |
| Phase 0-1 (prototype)                       | 50-700    | $0.1-3 k  | 1 ML engineer and 1 graphics engineer, about 1-2 months |
| Phases 2-4 (credible, without distillation) | 2-30 k    | $5-120 k  | 2-4 people, about 6-12 months                           |
| plus Option C distillation                  | +1-15 k   | +$2-60 k  | plus research time                                      |

## 8. Evaluation

All of this is proposal, built from cited metrics.

**Fidelity on held-out paired data (Option A).**

- PSNR, SSIM, LPIPS [Zhang 2018] and DISTS [Ding 2020] against the path-traced target.
- Report the gain over **NR off**, meaning the proxy passed straight through. That is the baseline that matters: the
  model has to beat doing nothing.
- Hold out whole scenes and assets, not just frames.

**Realism.**

- FID [Heusel 2017] and KID [Bińkowski 2018] against a held-out photo set (Options B and C) or the targets (Option A).
  Use KID for small sets.
- Do not use any metric that is also a training loss as a headline number.

**Temporal stability.**

- Warping error between consecutive outputs on non-occluded pixels [Lai 2018], and tLP and tOF [Chu 2018], on
  scripted camera paths.
- Measure **static-camera flicker** too: the temporal variance at a still frame as the noise seed changes. It isolates
  noise-driven instability. Report it for NR off as well; it is about zero there.
- Report the mean blend weight at rest and in motion, as frame.md does for Bistro.

**Faithfulness.**

- Edge and segmentation agreement with renderer ids (no hallucinated objects).
- Colour shift (ΔE in Oklab or CIELAB) at localTone = 0, which should be near-identity.
- A monotonic response of each conditioning scalar.

**Faces.** Identity preservation on synthetic or consented heads only (`docs/suggested-assets.md`). Check for
"glamorising" or uncanny drift. Press coverage reports exactly this criticism of NVIDIA's early previews (IEEE
Spectrum).

**User studies.**

- Two-alternative forced choice: NR on (our model) against NR off, on short clips. Also our model against path-traced
  ground truth where it exists.
- Follow ITU-R [BT.500](https://www.itu.int/rec/R-REC-BT.500) / ITU-T [P.910](https://www.itu.int/rec/T-REC-P.910)
  practice for viewing conditions, randomisation and rater screening.
- Ask raters separately about realism, faithfulness to the scene and flicker.

**Runtime.** It is unchanged by construction: same architecture, same kernels, same dispatches. A quick re-measure on
the exported model is still worth doing as a sanity check.

**Do not** compare against NVIDIA's NR outputs without legal clearance (section 9).

## 9. Legal and ethical constraints

This section raises issues. It is not legal advice. Each item needs review by counsel before data collection starts,
and again before release.

- **NVIDIA's model.** Do not use NVIDIA's weights, or images produced by DLSS 5 NR (for example, captures from shipping
  games), as training targets, distillation teachers or evaluation references. That applies even if someone is
  "entitled" to run them. NVIDIA's public DLSS SDK license ([github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS))
  forbids reverse engineering and derivative works of the SDK. Which terms govern the DLSS 5 NR weights and their
  outputs, and whether training on those outputs would be a derivative use, is **a question for legal review**. We do
  not assert an answer. OpenDLSS-NR's own README disclaims any NVIDIA weights or rights. A model we ship should be
  clean-room in its data.
- **The architecture.** Reimplementing a published network's structure for interoperability is what OpenDLSS-NR does
  under MIT. Whether that also covers training and distributing new weights for it, including any patent questions, is
  **for legal review**.
- **Naming.** Do not call the open model "DLSS". It is an NVIDIA trademark (see the README notice).
- **Dataset licenses.** Track the license of every asset and image, as `suggested-assets.md` does.
  - NC and ND terms are unsuitable for a model meant for public, possibly commercial use.
  - Whether **ShareAlike** (Hypersim is CC BY-SA 3.0) or **attribution** duties carry over to trained weights is
    unsettled. Get advice, and keep an attribution manifest either way.
  - Teacher models (Option C) have their own licenses and training-data provenance, and some forbid commercial use or
    training other models on their outputs.
- **People.** Face datasets mix licenses (FFHQ includes CC BY-NC images) and raise likeness, publicity-right and
  biometric-data issues (for example under GDPR). Prefer synthetic humans and scans with documented consent. Do not
  build any identity-embedding evaluation set from scraped faces. Note that the Lee Perry-Smith and "Marcus" scans in
  `suggested-assets.md` are scans of real people. Their CC licenses cover copyright, not necessarily likeness in
  generated imagery.
- **Bias and misuse.** A model that changes skin appearance can change perceived skin tone or age, so evaluate across
  skin tones. Publish a model card with the data sources, the known failure modes, and the fact that output is a
  generative re-rendering.

## 10. Phased plan

| phase                                         | work                                                                                                                                                                                                                                                          | milestone (exit criterion)                                                                                                                                                                              | main risks                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **0. Foundations** (2-4 weeks)                | The PyTorch module (section 5.1), the numerics-emulation mode, the importer and exporter (section 6.1), unit tests of the cast and the packing against `oracle.ts`. Measure throughput.                                                                       | The synthetic model imported into PyTorch matches the TSL network within the threshold of section 5.2. A random-init export loads in `NRModel` and runs bit-identically in the reference and TSL ports. | Undocumented details (QKV order, ViT specifics); a slow emulation mode        |
| **1. Prototype** (4-6 weeks)                  | A small Option A set (about 20-40 k frames, CC0 assets, three.js raster to Cycles), single frame, 256-512 crops, L1 + LPIPS, float training, then post-training FP8 export.                                                                                   | Beats NR off on held-out PSNR and LPIPS. The exported model runs in the demo. We measure how much post-training FP8 loses, which sizes the QAT work.                                                    | The realism ceiling; the FP8 cliff; the noise being ignored (acceptable here) |
| **2. Scale-up and conditioning** (2-3 months) | More data (hundreds of thousands of frames), an adversarial term, consistency losses from renderer attributes, conditioning-scalar training (section 2.3), multi-resolution and full-frame steps for the ViT. Optional Option B pilot with permissive photos. | Monotonic, near-identity-at-0 control by every scalar. A human 2AFC preference over NR off. FID and KID improving.                                                                                      | Hallucination; GAN instability; photo licensing                               |
| **3. Temporal** (1-2 months)                  | Clips, recurrent self-fed unrolling, warping and TecoGAN-style losses, blend-logit supervision (section 4).                                                                                                                                                   | Static-camera flicker and warping error within a stated margin of NR off. No ghosting on disocclusion tests.                                                                                            | Flicker from per-frame noise; exposure bias; memory cost of unrolling         |
| **4. FP8 QAT** (1 month, overlaps 2-3)        | Full fake-quant, magnitude reparameterisation and folding, a range regulariser, a low-learning-rate finish (section 5.3).                                                                                                                                     | The exported model's metrics are within about 0.3 dB / 0.01 LPIPS of the float model, and every boundary passes the health report.                                                                      | Underflow at large fan-in; saturation in the residual stream                  |
| **5. Export and ship**                        | Release export, fixtures, CI real-weights tests, a model card, an attribution manifest, legal sign-off, hosting (about 141 MiB; CDN and caching), a model switch in the demo.                                                                                 | Parity gates green on our weights. The model card is published. The demo ships with "open model" as its default.                                                                                        | License review blocking release; download size on the web                     |
| **6. Research (optional)**                    | Option C one-step distillation from an openly licensed teacher, with a noise-meaningful generative objective.                                                                                                                                                 | Preference over the Phase 5 model at equal temporal stability.                                                                                                                                          | Cost; teacher licensing; temporal stability under a generative prior          |

**Decision points.**

- After Phase 1, decide whether the prototype's quality over NR off is enough to justify Phases 2-4.
- After Phase 0, if emulation fidelity is poor, decide whether to write a faithful F13 emulator before going on.
- Before Phase 2, decide whether to read the DLSS 5 report in full. It could change the objective.

## 11. References

**Project sources**

- OpenDLSS-NR by maan (MIT), pinned at `9d08f41`:
  [README](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/README.md),
  [docs/network.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/network.md),
  [numerics.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/numerics.md),
  [weights.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/weights.md),
  [frame.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/frame.md),
  [execution.md](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/docs/execution.md). The same files are in this
  repository at [`reference/OpenDLSS-NR`](../reference/OpenDLSS-NR) (a git submodule).
- This repository:
  - [`packages/three-dlss-nr/src/model/layouts.ts`](../packages/three-dlss-nr/src/model/layouts.ts): the 153 records
    and their regions.
  - [`model/manifest.ts`](../packages/three-dlss-nr/src/model/manifest.ts): manifest validation.
  - [`model/Model.ts`](../packages/three-dlss-nr/src/model/Model.ts): the loader and the |w| <= 9 bound.
  - [`synthetic/`](../packages/three-dlss-nr/src/synthetic): the generator, gains and calibration gate.
  - [`numerics/oracle.ts`](../packages/three-dlss-nr/src/numerics/oracle.ts): the E4M3 and f16 conversions and the
    packing indices.
  - [`test/browser/run-network-parity-chrome.mjs`](../packages/three-dlss-nr/test/browser/run-network-parity-chrome.mjs).
  - [`docs/suggested-assets.md`](suggested-assets.md).
  - The port's design notes (`three-dlss-nr-design.md`, kept beside the repository). §5.1 covers the synthetic model
    directory.

**NVIDIA DLSS 5**

- [DLSS 5: Generative Neural Rendering](https://research.nvidia.com/labs/adlr/DLSS5/), project page, published
  2026-09-01. The [report PDF](https://research.nvidia.com/labs/adlr/DLSS5/files/DLSS5_Report.pdf) returned 403 to
  automated fetches when this was written.
- NVIDIA Developer Blog,
  [What's New for Game Developers: DLSS 5 with 3D-Guided Neural Rendering…](https://developer.nvidia.com/blog/whats-new-for-game-developers-dlss-5-with-3d-guided-neural-rendering-nvidia-ace-updates-and-new-rtx-kit-capabilities/).
- IEEE Spectrum, [Neural Rendering Redefines Photo-real Faces in Games](https://spectrum.ieee.org/neural-rendering-nvidia-dlss-5).
- NVIDIA DLSS SDK license: [github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS).

**Literature (each verified to exist)**

- [Richter 2021] S. R. Richter, H. A. AlHaija, V. Koltun, _Enhancing Photorealism Enhancement_,
  [arXiv:2105.04619](https://arxiv.org/abs/2105.04619); code: [isl-org/PhotorealismEnhancement](https://github.com/isl-org/PhotorealismEnhancement).
- [Xiao 2020] L. Xiao et al., _Neural Supersampling for Real-time Rendering_, SIGGRAPH 2020,
  [research.facebook.com](https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/).
- [Lai 2018] W.-S. Lai et al., _Learning Blind Video Temporal Consistency_, [arXiv:1808.00449](https://arxiv.org/abs/1808.00449).
- [Chu 2018] M. Chu et al., _Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation_
  (TecoGAN), [arXiv:1811.09393](https://arxiv.org/abs/1811.09393).
- [Liu 2021a] Z. Liu et al., _Swin Transformer_, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).
- [Liu 2021b] Z. Liu et al., _Swin Transformer V2_ (scaled cosine attention), [arXiv:2111.09883](https://arxiv.org/abs/2111.09883).
- [Liang 2021] J. Liang et al., _SwinIR_, [arXiv:2108.10257](https://arxiv.org/abs/2108.10257).
- [Wang 2021] Z. Wang et al., _Uformer: A General U-Shaped Transformer for Image Restoration_,
  [arXiv:2106.03106](https://arxiv.org/abs/2106.03106).
- [Karras 2023] T. Karras et al., _Analyzing and Improving the Training Dynamics of Diffusion Models_ (EDM2),
  [arXiv:2312.02696](https://arxiv.org/abs/2312.02696).
- [Micikevicius 2022] P. Micikevicius et al., _FP8 Formats for Deep Learning_, [arXiv:2209.05433](https://arxiv.org/abs/2209.05433).
- [Esser 2019] S. K. Esser et al., _Learned Step Size Quantization_, [arXiv:1902.08153](https://arxiv.org/abs/1902.08153).
- [DeepSeek-AI 2024] _DeepSeek-V3 Technical Report_ (FP8 training at scale), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
- NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).
- [Bachlechner 2020] _ReZero is All You Need_, [arXiv:2003.04887](https://arxiv.org/abs/2003.04887).
- [Zhang 2019] _Fixup Initialization_, [arXiv:1901.09321](https://arxiv.org/abs/1901.09321).
- [Touvron 2021] _Going deeper with Image Transformers_ (LayerScale), [arXiv:2103.17239](https://arxiv.org/abs/2103.17239).
- [Ledig 2017] _Photo-Realistic Single Image Super-Resolution Using a GAN_ (SRGAN), [arXiv:1609.04802](https://arxiv.org/abs/1609.04802).
- [Zhu 2017] _CycleGAN_, [arXiv:1703.10593](https://arxiv.org/abs/1703.10593).
- [Park 2020] _Contrastive Learning for Unpaired Image-to-Image Translation_ (CUT), [arXiv:2007.15651](https://arxiv.org/abs/2007.15651).
- [Rombach 2022] _High-Resolution Image Synthesis with Latent Diffusion Models_, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).
- [Zhang 2023] _Adding Conditional Control to Text-to-Image Diffusion Models_ (ControlNet), [arXiv:2302.05543](https://arxiv.org/abs/2302.05543).
- [Sauer 2023] _Adversarial Diffusion Distillation_, [arXiv:2311.17042](https://arxiv.org/abs/2311.17042).
- [Yin 2023] _One-step Diffusion with Distribution Matching Distillation_, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828).
- [Parmar 2024] _One-Step Image Translation with Text-to-Image Models_, [arXiv:2403.12036](https://arxiv.org/abs/2403.12036).
- [Zhang 2018] _The Unreasonable Effectiveness of Deep Features as a Perceptual Metric_ (LPIPS), [arXiv:1801.03924](https://arxiv.org/abs/1801.03924).
- [Ding 2020] _Image Quality Assessment: Unifying Structure and Texture Similarity_ (DISTS), [arXiv:2004.07728](https://arxiv.org/abs/2004.07728).
- [Heusel 2017] _GANs Trained by a Two Time-Scale Update Rule…_ (FID), [arXiv:1706.08500](https://arxiv.org/abs/1706.08500).
- [Bińkowski 2018] _Demystifying MMD GANs_ (KID), [arXiv:1801.01401](https://arxiv.org/abs/1801.01401).
- [Teed 2020] _RAFT: Recurrent All-Pairs Field Transforms for Optical Flow_, [arXiv:2003.12039](https://arxiv.org/abs/2003.12039).
  Useful for estimating flow on photo or video data that has no engine motion vectors.
- [Roberts 2021] _Hypersim_, [arXiv:2011.02523](https://arxiv.org/abs/2011.02523); license per
  [apple/ml-hypersim](https://github.com/apple/ml-hypersim) (CC BY-SA 3.0).
- [Richter 2016] _Playing for Data: Ground Truth from Computer Games_, [arXiv:1608.02192](https://arxiv.org/abs/1608.02192).
- FFHQ: [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) (license section).
- ITU-R [BT.500](https://www.itu.int/rec/R-REC-BT.500) and ITU-T [P.910](https://www.itu.int/rec/T-REC-P.910).
