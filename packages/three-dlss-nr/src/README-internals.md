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
