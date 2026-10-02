# three-dlss-nr internals: writing and testing a kernel

three-dlss-nr ports [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by maan (MIT, pinned at `9d08f41`) to
three.js TSL. The spec for every kernel is the reference WebGPU port (`reference/OpenDLSS-NR/ports/browser-webgpu`),
byte for byte; the design (`three-dlss-nr-design.md`, Appendix A) lists the byte-level rules per kernel. This page
covers the foundation the kernels are built on.

## Layout

| file                   | what                                                                                                                                                                                                                                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`             | `NRTensor`, `FP8Matrix`, `HalfVector`, `F32Vector`, `F16Matrix`, `NRKernel`, `GemmSpec` and the per-kernel spec/buffer types: the contract between chunks                                                                                                              |
| `tensors.ts`           | `createTensor`, `NRTensors` (the reference's `Tensors.allocate`: rows padded to 64, zero-filled, keyed), `createHalfVector`, `createF32Vector`, `attributeFromBytes`, `writeBuffer`, `fillBuffer`, `readBuffer`                                                        |
| `geometry.ts`          | port of `geometry.js`: `geometryFromValid`, the four fused layouts, `windowPhase`, `WindowPhases`, `grid1d`                                                                                                                                                            |
| `device.ts`            | `createNRDevice` (adapter-max limits, `shader-f16` when present), `checkNRDeviceLimits`, `createNRRenderer`                                                                                                                                                            |
| `numerics/oracle.ts`   | TS oracle: port of `numerics.js` plus the production publications (`publishE4CodeGemm`, `fp8DomainBitQuant`, `exactE4Code`, `siluE4CodeTable`) and the index maps (`packedInputIndex`, `packedWeightIndex`, `packedF16WeightIndex`, `tiledToken`, `inverseTiledToken`) |
| `tsl/numerics.ts`      | TSL port of `numerics.wgsl`: layout-`Fn` helpers `nrPow2` ... `nrFixedToF16`, `nrPublishE4CodeGemm`, inline `nrFdpa16` / `nrFdpaF16x8`                                                                                                                                 |
| `tsl/packed.ts`        | typed literals `u` / `i` / `f` / `fBits`, branch-free `pick`, packed access `loadE4`, `loadHalfBits`, `loadHalf`, `loadF32`, `byteOf`, `packWord4`, `packHalfPair`                                                                                                     |
| `tsl/KernelBuilder.ts` | `kernel()`, `runKernels()`, `kernelWGSL()`, `foldedGroupY()`                                                                                                                                                                                                           |

Test helpers live in `packages/three-dlss-nr/test/`: `gpu.ts` (device + renderer per file, `runPerElement`),
`compare.ts` (the reference's `compareCodes` / `compareFloats`, `diffArrays`, `fillSentinel`), `wgsl.ts`
(`normalizeWGSL`, `expectIntegerComparisons`, `expectNoApproximations`), `oracle/gemm.ts` (CPU GEMM oracles),
`reference/refKernels.ts` (single reference dispatches), `reference/refNetwork.ts` (the whole reference network),
`setup/fetchShim.ts` (`file:` and `synthetic:` fetch; `registerSyntheticFiles`). Reference modules are imported as
`@ref/<file>.js` (vitest alias; typed `any` by `test/ref-shims.d.ts`).

## Writing a kernel

```ts
import { If, localId, workgroupId } from 'three/tsl';
import { grid1d } from '../geometry.js';
import type { NRKernel, NRTensor } from '../types.js';
import { kernel } from '../tsl/KernelBuilder.js';
import { nrEncodeE4m3, nrF16Bits, nrRoundF16 } from '../tsl/numerics.js';
import { f, loadHalf, packWord4, u } from '../tsl/packed.js';

export function createHalveToE4(input: NRTensor, output: NRTensor, label: string): NRKernel {
  const count = input.rows * input.channels; // a multiple of 4
  return kernel({
    label, // the reference's dispatch label, e.g. 'block 5 expert expand'
    kind: 'halve', // the reference's entry point; also the compute node's name
    workgroupSize: [64],
    dispatch: grid1d(count / 4), // explicit [x, y, z] workgroup counts, never a count
    inputs: { source: input }, // bound read-only; WGSL name nr_source
    outputs: { target: output }, // bound read_write; recorded in `writes`
    body: ({ source, target }) => {
      const quad = workgroupId.x
        .add(workgroupId.y.mul(u(65535)))
        .mul(u(64))
        .add(localId.x)
        .toVar();
      If(quad.lessThan(u(count / 4)), () => {
        const code = (k: number) =>
          nrEncodeE4m3(nrF16Bits(nrRoundF16(loadHalf(source, quad.mul(u(4)).add(u(k))).mul(f(0.5)))));
        target.element(quad).assign(packWord4(code(0), code(1), code(2), code(3)));
      });
    },
  });
}
```

- All buffers are `array<u32>` views; a byte tensor is written a whole word (4 values) per invocation and a half
  tensor a whole word (2 values). There is no 8- or 16-bit store: thread mappings must own whole words.
- Shapes, strides, flags are JS numbers baked into the node graph (the reference's `override`s). Kernels that differ
  only in buffers produce identical WGSL and share one compiled program; keep labels out of the generated code.
- A kernel's body runs lazily, when three first builds the node; buffers must be declared in `inputs` / `outputs`.
- Run a frame with `runKernels(renderer, kernels)` (one compute pass, in order, validation error scope) or
  `renderer.compute(kernels.map((k) => k.node))`. Read with `readBuffer(renderer, tensor)`.
- Look at the generated code with `kernelWGSL(renderer, kernel)`; snapshot `normalizeWGSL(...)` to catch drift.

## Comparing against the reference

```ts
const gpu = await createGpuTestContext(); // our device, shader-f16 when available
const ref = await RefKernels.create(gpu.device); // the reference port on the same device
const reason = ref.unavailableReason('gemm_fp8'); // e.g. 'needs shader-f16, which this device lacks'
if (reason) return context.skip(reason); // use it.for(...)(name, async (args, context) => ...)

const input = ref.tensor('in', rows, k, 'e4', bytes);
const out = ref.tensor('out', rows, n, 'e4');
ref.fill(out, SENTINEL_BYTE);
ref.gemm({ input, weights: ref.fp8Matrix({ bytes: w, k, n, scales }), output: out, rows, k, n, label });
await ref.run();
const expected = await ref.read(out); // validBytes, like the reference's readback

const ours = createTensor('out', rows, n, 'e4');
fillSentinel(ours); // R6: an unwritten output stays 0xCD on both sides
await runKernels(gpu.renderer, [createGemmFp8(spec, { input: oursIn, weights, output: ours })]);
const verdict = compareCodes(await readBuffer(gpu.renderer, ours), expected);
expect(verdict.verdict).toBe('bit-exact');
```

`RefKernels` also records `gemmF16`, `op` (ops.wgsl entry points), `windowAttention` and `vit` dispatches, and exposes
the recorder for anything else. Where the reference cannot run (see below), compare against the CPU oracles
(`test/oracle/gemm.ts`, `numerics/oracle.ts`); `test/reference/refKernels.gpu.test.ts` ties those oracles to the
reference on every device where the reference runs.

### Where the reference runs

The reference's FP8 GEMM, window attention and SiLU-table builders need `shader-f16`. Whether Dawn in Node exposes it
depends on the backend: **not on the Windows dev machine** (RTX 3060 Ti: D3D12 lacks DXC in the `webgpu` package's
Dawn build, and NVIDIA's Vulkan driver does not qualify); lavapipe (CI) is expected to, which the CI device report
will confirm. In addition, **FXC (D3D12 without DXC) fails to compile the reference's `normal_exponent`
(`bitcast<u32>(abs(x))`) with `E_FAIL`**, so the reference's `gemm_f16.wgsl`, `vit.wgsl` and FDPA self-test cases do not
build on D3D12 here; `ops.wgsl` does. On Dawn's Vulkan backend (`DLSS_NR_DAWN_BACKEND=vulkan`, NVIDIA) those compile
and match the oracles. `refKernels.gpu.test.ts` prints a device report of what is available.

Our TSL must compile under FXC too (the default local backend): test every kernel on D3D12.

**lavapipe (CI) has `shader-f16` but is not a trustworthy f16 reference.** Mesa llvmpipe (25.2, LLVM 20) folds the
round trip `f32(f16(x))` into `x`. The reference spells every half rounding that way: `round_accumulator`, the end
of each FP8 GEMM FDPA group (`vec4<f32>(vec4<f16>(sums * scale))`), and the SiLU table builder. On llvmpipe the half
accumulator therefore stays an unrounded f32 between groups, and a real f16 rounding happens only where a value is
stored as f16 or bitcast. In CI run 37030840217, 24% of the reference FP8 GEMM's raw half outputs were one half ulp
off. A CPU model of that folding reproduces every listed CI difference. The same inputs in Chrome on an RTX 3060 Ti
(D3D12 + DXC, real f16) equal `oracleGemmFp8` bit for bit on all seven cases (`test/browser/run-fp8-gemm-chrome.mjs`,
one-off and not part of any test run). So `RefKernels` reports `gemm_fp8` and `window_attend` unavailable on
llvmpipe (`SOFTWARE_F16_REASON`; override it with `DLSS_NR_TRUST_SOFTWARE_F16=1`). The local gate for reference
parity of the f16 kernels is real hardware in a browser:

```sh
node packages/three-dlss-nr/test/browser/run-fp8-gemm-chrome.mjs            # Chrome stable; --chrome <path>, --headed
```

Our TSL never relies on f16 hardware. It rounds with `nrRoundF16` on bit patterns, so it gives the same bytes on
lavapipe and on hardware.

## TSL pitfalls and the helper that avoids each

| #         | pitfall (three 0.186)                                                                                                                                                                                                                                                                | avoid it with                                                                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1        | `a.op(b)` casts both operands to `a`'s type; a bare JS number is `float`, so `intNode.mul(0.5)` emits `i32(0.5)`                                                                                                                                                                     | typed literals `u()`, `i()`, `f()`; never a bare number as an operand                                                                                      |
| R2        | comparing mismatched types compares as f32 (`f32(i) < f32(rows)`), wrong above 2^24                                                                                                                                                                                                  | `x.lessThan(u(rows))`; assert with `expectIntegerComparisons(kernelWGSL(...))`                                                                             |
| R3        | float literals print as the shortest JS repr (exact only for f32 values), `float(-0)` prints `0.0`, a negative `uint` prints `0u`, `int` is `Math.round`ed                                                                                                                           | `f()` asserts f32-exact and rejects -0 (use `fBits(0x80000000)`); `u()` / `i()` assert range                                                               |
| R4        | backends may contract `a*b+c` into fma                                                                                                                                                                                                                                               | only write `a*b+c` where `a*b` is exact in f32; no `mix`, `smoothstep`, `pow`, `exp2`, `log2`, `inverseSqrt` in published paths (`expectNoApproximations`) |
| R5        | `1/sqrt(x)` and `1/x` must stay that spelling (matched by outcome)                                                                                                                                                                                                                   | `f(1).div(sqrt(x))`, never `inverseSqrt`                                                                                                                   |
| R6        | nodes not reachable from an assignment are dropped; an output never written reads as whatever was there                                                                                                                                                                              | outputs declared in `kernel({ outputs })` become `writes`; tests prefill them with `fillSentinel` / `ref.fill(t, SENTINEL_BYTE)` and compare whole buffers |
| R7        | two `storage()` nodes on one attribute are two bindings; two writable bindings of one buffer get the whole command buffer rejected                                                                                                                                                   | `kernel()` creates one storage node per attribute, binds inputs read-only, throws on a buffer declared twice; `runKernels` validates                       |
| R8        | Tint folds literal subexpressions at abstract precision                                                                                                                                                                                                                              | precompute constants in JS with `Math.fround` and pass them through `f()`                                                                                  |
| R9        | signed zeros differ by kernel (GEMM: -0 -> 0x00; window/ops/ViT: keep 0x80; FDPA results +0)                                                                                                                                                                                         | one publication function per reference kernel (`nrPublishE4CodeGemm` vs `nrEncodeE4m3(nrF16Bits(x))`); explicit canonicalization, never `0.0 + x`          |
| R10       | backends flush f32 subnormals (D3D12 does)                                                                                                                                                                                                                                           | arithmetic never forms f32 subnormals; tests skip f32-subnormal inputs where the reference would flush too                                                 |
| R12       | `instanceIndex` comes from `globalId` and `numWorkgroups`; `Loop` counters default to `int`                                                                                                                                                                                          | index from `workgroupId` / `localId`, `foldedGroupY()` past 65535 row groups; `Loop({ type: 'uint' })`                                                     |
| R14 (new) | bare `select(c, a, b)` emits `if/else` and re-generates each operand's subtree in both branches: nested selects grow exponentially (an 8-term FDPA became 3500 lines and FXC failed); `select(...).uniformFlow()` emits WGSL `select` but a second use reads an unassigned temporary | `pick(cond, ifTrue, ifFalse)` = `select(...).uniformFlow().toVar()`                                                                                        |
| R15 (new) | the compute node's name is written into the WGSL (`// flow -> name`), defeating program sharing                                                                                                                                                                                      | `kernel()` names the node by `kind`; the label stays in `NRKernel.label`                                                                                   |
| R16 (new) | FXC (D3D12 without DXC) fails with `E_FAIL` on `bitcast<u32>(abs(x))` (and on very large functions)                                                                                                                                                                                  | read exponent fields straight off the bits (`nrNormalExponent`); keep kernels small with layout `Fn`s and `pick`                                           |

Binding names: `kernel()` names each binding `nr_<declared name>` instead of three's `NodeBuffer_<id>`.

## The graph and the network

`graph/Graph.ts` (`NRGraph`) is the port of `graph.js` `record` (lines 354-580). It builds 451 `NRKernel`s per valid
size. Their order, labels, kinds and workgroup counts equal a record-only run of the reference graph
(`graph/Graph.test.ts` checks this at 64x64, 300x500 and 512x512, through `test/reference/refModel.ts`
`recordGraphRequests`). Its tensor labels, shapes and formats equal the reference's too, so `readTensor(label)`
names the same tensor on both sides.

- Weights come from `NRModel`. The GEMM skip scales, which the reference reads at a byte offset into the stage
  buffer, come from `auxVector(tensor, offset, n)`. The post blend's two scales are one `auxPair`.
- Boundary captures (`captureBoundaries`) are `copy_words` kernels inside the one pass. There are 79 of them:
  `block-0` to `block-69` and the nine `transition-a-b` / `pooled-a-b`. A capture is not a dispatch.
  `NRGraphPass.index` is the index of the dispatch it follows, so `run({ until })` runs the first `until`
  dispatches and their captures, with the same index as the reference's recorder.
- `graph/attention.ts` is the graph's only link to the attention kernels. `windowQueriesFor(device)` picks 16
  queries per workgroup on devices that cannot run 512 invocations. The bytes are the same; the dispatch size is
  not.

`NRNetwork.ts` implements `NRBackend` (`id: 'tsl'`, factory `tslBackend`).

- A frame is one `renderer.compute(nodes)`.
- `create` compiles every node up front, one at a time, because three's progress callback needs `ProgressEvent`,
  which Node lacks.
  - Expect about 4-5 minutes on D3D12/FXC in Node at 64x64, most of it FXC.
- A model passed as an `NRModel` is borrowed and never disposed. Load it once and share it across resizes.
- `writeFeatures` writes the GPU buffer with `queue.writeBuffer` once it exists.

### Full-network parity (real Chrome)

`node packages/three-dlss-nr/test/browser/run-network-parity-chrome.mjs [--sizes 64x64,512x512] [--golden] [--stats]`
runs the reference `Network` (its own WGSL, fetched from the submodule) and `NRNetwork` on one device in one page
(`networkParityPage.ts`), with the synthetic model and `syntheticFeatures`. It compares all 79 boundaries, the three
post tensors and the head, byte for byte, and checks that a second run of each gives the same bytes. It also prints
the reference's per-boundary statistics, which are the synthetic-weights calibration gate.

- `--golden` writes our digests to `test/network/goldenDigests.ts`.
- `network.synthetic.gpu.test.ts` then checks the network against those digests in Node on any device, including CI
  lavapipe. The TSL port never uses f16 hardware.
- `--bisect N` runs both networks truncated after dispatch N and compares every tensor label both allocate.
- `--stats-only` runs the reference alone and checks the calibration gate. It is fast because nothing is compiled
  through TSL. Use it when changing `synthetic/gains.ts`, then re-pin the stage hashes (`generate.test.ts`) and
  rerun with `--golden`. The skip-scale ranges are per family (general, ViT, decoder window blocks): the residual
  stream's gain per block is steep in the mean scale.
- Results on the dev machine (RTX 3060 Ti, Chrome stable, D3D12 + DXC): at 64x64 and at 512x512, 83 of 83 tensors
  are bit-exact (79 boundaries, `post merge`, `post merge raw`, `post block raw`, head), repeat runs are identical,
  and 79 of 79 boundaries are in calibration range.
- In Node, `network.synthetic.gpu.test.ts` runs 64x64 by default. 512x512 needs `NR_FULL=1`: its single
  451-dispatch command buffer can trip the Windows driver timeout (TDR) on D3D12/FXC when the GPU is busy.
- `network.real.gpu.test.ts` runs the reference's own `loadFixture` / `runParity` on `NRNetwork`. It needs
  `NR_WEIGHTS` (a model directory) and `NR_FIXTURES` (a fixture directory, or a directory of them).

## Backends: the native TSL port and the reference shim

`src/backend/NRBackend.ts` defines `NRBackend` / `NRBackendFactory`. That is the network API of design chunk E, plus
`id`, `requirements`, GPU-resident `features` / `head` tensors, `memory`, and a timing result from `run`. The native
`NRNetwork` implements it unchanged (`class NRNetwork implements NRBackend`). Export a factory named `tslBackend` from
the index and the benchmark and parity page pick it up. `test/backendConformance.ts` (`backendProblems`,
`factoryProblems`) is the shared conformance check.

`src/reference-backend/` (entry point `three-dlss-nr/reference-backend`) is the **shim**: OpenDLSS-NR's reference
WebGPU port, its JS and WGSL byte for byte, driven on `renderer.backend.device`.

- **Bundling.** `scripts/bundle-reference.mjs` copies into `vendor/opendlss-nr/` (git-ignored, published via
  `files`) every module reachable from `src/network.js`, plus the WGSL as string constants in `shaders.js`. Each file
  gets a header naming the upstream file, the commit and the MIT license, and `LICENSE` and `SOURCE.json` (SHA-256 per
  file) go beside them. The script runs in `build`, `pnpm tsc` and `pnpm test:gpu`, and `--check` reports drift. It
  refuses a submodule that is not at the pinned commit. `bundle.test.ts` checks the vendored files against
  `git show <commit>:...`.
- **Orchestration.** `ReferenceWgslBackend.create` is the upstream `Network.create` without its `fetch` of the WGSL.
  The object it builds is an upstream `Network` instance, so `run`, `readHead` and `readBoundary` are the upstream
  methods. A model given as a URL loads through the upstream `Model.load`. An in-memory model (manifest plus stage
  bytes) is uploaded with the same steps.
- **Shared buffers.** `features` and `head` are `createTensor` storage attributes. `sharedTensorBuffer` has three
  create their GPU buffers up front (`renderer.backend.createStorageAttribute`) and hands them to the upstream graph
  under the keys the graph allocates them with. three's attribute code never creates a second buffer for an attribute
  that already has one. So a TSL frame node that binds `backend.features` or `backend.head` writes or reads the same
  buffer the WGSL graph uses. Writing `features` from the CPU goes through `queue.writeBuffer`, not `needsUpdate`. Do
  not set `needsUpdate` on these attributes: three would upload their stale CPU copy over the GPU data.
- **Frame adapter.** `ReferenceFrame` records the upstream demo's `input_features` / `compose` / history swap exactly
  as `production-pipeline.js` does, but takes each frame from three textures on the GPU:
  - A small pack pass writes the upstream `scene` buffer (RGBA16F halves) and `motion` buffer with `flip_y = 0`.
    `motion` is the uv offset to the previous position, y down, which is `(-v.x/2, v.y/2)` of three's NDC `velocity`.
  - A present pass unpacks the upstream's bgra8 canvas image into a three `StorageTexture` (`output`, rgba8unorm,
    display-ready sRGB bytes).
  - History works as upstream: it is invalid on the first frame, after `resetHistory()` or `reset: true`, and after a
    rebuild. The noise seed is the frame count.
  - Render targets: `RenderTarget(w, h, { count: 2, type: HalfFloatType, samples: 0 })`. Name the textures `output`
    and `velocity` for `mrt({ output, velocity })`, and call `setClearColor('velocity', black, 0)` so the background
    has no motion.
- **Timing** (`NRFrameTimer` in `src/backend/timing.ts`). Both backends bracket the frame on the queue with two empty
  compute passes that carry timestamp writes (`timestamp-query`). Both also take wall time from the first submit to
  `onSubmittedWorkDone`. The brackets do not depend on how a backend encodes its frame, so TSL and WGSL are measured
  the same way. The first `run` of a backend validates its frame inside an error scope and reports wall time only.

### Real-browser checks (not part of `pnpm test`)

Dawn in Node has no `shader-f16` on the Windows dev machine, so the shim runs in Chrome (D3D12 + DXC).
`test/browser/chrome.mjs` is a DevTools harness with no dependencies. It uses Chrome stable, else Playwright's
`%LOCALAPPDATA%\ms-playwright\chromium-*`. `test/browser/shimPage.ts` is the page.

- `node packages/three-dlss-nr/test/browser/run-shim-parity-chrome.mjs [--model <dir>] [--size 128x128]` compares the
  shim against the upstream port running standalone, with its own device, WGSL fetched from the submodule and the
  upstream `Network.create`.
  - Network: the same features in, then the head compared bitwise and every boundary compared.
  - Frames: three rendered frames with a moving camera. It compares the packed inputs, the features (noise lanes
    included), the head, the history and the presented image. The standalone side receives each frame through a CPU
    readback, as the upstream demo does.
- `node scripts/bench-backends.mjs [--model <dir>] [--sizes 512x512,1280x720]` reports, per backend and size, GPU and
  wall ms (min / median), create time, first-frame time and memory. Run it on an idle GPU.

`--model` defaults to `$NR_MODEL_DIR`. Without it, chunk A's generator writes a synthetic model once to
`node_modules/.cache/three-dlss-nr/synthetic-model`.

### Notes for the website (chunk G) and the fidelity suite (chunk H)

- **Switching at runtime.**
  - Keep one renderer, created on `createNRDevice()` so that `shader-f16` is enabled when the adapter has it.
  - List `[tslBackend, referenceWgslBackend]`. Disable a backend whose `unavailableReason(renderer)` is non-null and
    show that reason, which names the fix.
  - To switch, `dispose()` the current backend and `create` the other one at the same size.
  - Both backends read the same `features` layout and write the same `head`, so the frame code is shared. Your TSL
    input-features pass writes `backend.features` and your compose pass reads `backend.head`, with no copies.
  - To show the reference's own frame instead (its compose, display transform and history), use `ReferenceFrame` and
    draw `frame.output` with no tone mapping.
  - Reset history on every switch: the history buffers differ between the two paths.
- **Compile and warm-up.**
  - Creating the reference backend compiles about 200 specialised GEMM pipelines and about a dozen window-attention
    pipelines (`createComputePipelineAsync`, bounded concurrency), and records 451 dispatches.
  - Pipelines are cached per device and per override set, so a second `create` at the same size costs much less.
    Show the `onProgress` messages while it runs.
  - The first `run` validates the frame.
  - Load the weights once with `loadReferenceModel` and pass the result to every `create`. Otherwise each resize
    re-uploads about 141 MiB. Dispose the model separately.
  - The TSL backend loads its own weights, so switching keeps two copies on the GPU unless you dispose the one you
    are not using.
- **Memory** (reported by `backend.memory`). The reference holds:
  - the stage buffers (the whole model, about 141 MiB) plus its re-laid-out f16 matrices, priors and scales;
  - its activation tensors: many hundreds of MiB at 1280x720, more with `captureBoundaries`;
  - the SiLU and weight tables, kept per device;
  - with `ReferenceFrame`, about 20 bytes per pixel of frame buffers.

  Cap the demo's internal resolution as the design says: at most 1280x720 valid.

- **Fidelity suite** (`packages/fidelity-suite`, published at `/parity/`; see its README).
  - Three renderers per scene: `opendlss-nr` (standalone reference), `three-dlss-nr-tsl` and `three-dlss-nr-shim`,
    on the same CPU-built features from a three.js render.
  - To compare quality, run `tsl` and `reference-wgsl` through `NRBackend` on the same `writeFeatures` input, then
    compare `readHead` and `readBoundary(name)`. Boundary names are the reference's.
  - The shim needs `shader-f16`. In Node it only runs where Dawn exposes that feature, and lavapipe's f16 results are
    untrusted ("[B: lavapipe]" in the design). Render the committed results in Chrome on the dev GPU with the harness
    above.

## The integration pass (`integration/DlssNrPass.ts`, design chunk G)

- One frame: the scene into an RGBA16F MRT target (`textures[0]` colour named `output`, `textures[1]` named
  `velocity`, velocity cleared to 0), then `inputFeatures[p]` (`renderer.compute`), `backend.run()` (its own submit,
  awaited), then `compose[p]` plus a small unpack of the reference's bgra8 `image` into an rgba8unorm
  `StorageTexture`, then the present quad. `p = frameIndex & 1` selects the `NRHistory` ping-pong.
- Both backends use the same frame kernels: the features and head are three storage attributes bound to the
  backend's own buffers (`sharedTensorBuffer` for the shim), so nothing is copied. Compose writes the reference's
  `image` (display transform applied on the GPU, exactly as `frame.wgsl`), so the presented bytes are the
  reference's whenever D's kernels are byte-identical to `frame.wgsl` (D3D12: yes; NVIDIA Vulkan: rare one-step
  differences in the reprojected history). The quad samples that texture with bilinear filtering to the canvas size.
- NR off is drawn by the quad as `nrDisplayTransform(scene)` in a fragment shader: the same operator, but not
  bit-identical to the compute version (fma contraction), which is irrelevant for viewing.
- `DlssNrExternalFrame` is the escape hatch for a backend that records its own frame (`ReferenceFrame`); its
  `output` must be display-ready rgba8 bytes.
- The blend scale comes from `backend.blendScale` (shim) or `backend.model.blendScale()` (TSL port), else the
  `blendScale` option.
- `render()` waits for the network on the GPU (as the reference demo does), so the loop runs at the network's
  frame rate; a call while one is in flight returns null.
- Fidelity suite: `DlssNrPass` exposes `renderTarget` (the rendered colour / velocity) and `outputTexture` (the
  presented NR image); render into an RGBA8 target with `renderer.setRenderTarget(target)` before `render()` to read
  the presented frame back (the GPU test does this).
