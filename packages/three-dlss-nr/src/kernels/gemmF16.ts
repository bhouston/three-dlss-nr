// The f16 GEMM: the network's input adapter (16 -> 32) and head (32 -> 4), the two matrices with half operands.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). A literal port of
// ports/browser-webgpu/shaders/gemm_f16.wgsl as graph.js `Graph.gemmF16` dispatches it: one invocation per (row, four
// output columns), workgroups of 64 over `grid1d(rows * n / 4)` folded past 65535; per output, `ada_f16_fdpa8` over K
// in steps of 8 from a zero accumulator (24 fractional bits, exact i32 sum, exact `fixed_to_f16`), published as f32,
// as packed halves, and/or as `encode_e4m3(f16_bits(v))` (which keeps -0 as 0x80, unlike the FP8 GEMM). The products
// of two halves are exact in f32 (22 significant bits), so the f32 arithmetic of `nrFdpaF16x8` equals the reference's.

import { If, Loop, localId, workgroupId } from 'three/tsl';

import { grid1d } from '../geometry.js';
import { kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { nrEncodeE4m3, nrF16Bits, nrFdpaF16x8 } from '../tsl/numerics.js';
import { bitsOf, f, loadHalf, packHalfPair, packWord4, u, type TSLNode } from '../tsl/packed.js';
import type { GemmF16Buffers, GemmF16Spec, NRKernel, NRTensor } from '../types.js';

const MAX_GROUPS = 65535;

/**
 * One f16 GEMM dispatch: `out[row][column] = sum_k input[row][k] * weights[k][column]` through the Ada f16 FDPA step.
 * `k % 8 == 0`, `n % 4 == 0`; every given output shares one row stride (`outputF32 ?? outputF16 ?? output`).
 */
export function createGemmF16(spec: GemmF16Spec, buffers: GemmF16Buffers): NRKernel {
  const { rows, k, n, label } = spec;
  const { input, weights, output, outputF16, outputF32 } = buffers;
  if (!Number.isInteger(rows) || rows < 1) throw new RangeError(`${label}: rows ${rows}`);
  if (k < 8 || k % 8) throw new RangeError(`${label}: K ${k} is not a positive multiple of 8`);
  if (n < 4 || n % 4) throw new RangeError(`${label}: N ${n} is not a positive multiple of 4`);
  if (weights.k !== k || weights.n !== n || weights.paddedN < n) {
    throw new Error(`${label}: a ${weights.k}x${weights.n} (padded ${weights.paddedN}) matrix for ${k}x${n}`);
  }
  if (input.format !== 'f16') throw new Error(`${label}: input must be an f16 tensor`);
  if (input.channels < k) throw new Error(`${label}: input stride ${input.channels} < K ${k}`);
  const target = outputF32 ?? outputF16 ?? output;
  if (!target) throw new Error(`${label}: no output`);
  const stride = target.channels;
  const check = (tensor: NRTensor | undefined, format: string) => {
    if (!tensor) return;
    if (tensor.format !== format) throw new Error(`${label}: ${tensor.label} must be ${format}`);
    if (tensor.channels !== stride) throw new Error(`${label}: every output must share the row stride ${stride}`);
    if (tensor.rows < rows) throw new Error(`${label}: ${tensor.label} has fewer than ${rows} rows`);
  };
  check(output, 'e4');
  check(outputF16, 'f16');
  check(outputF32, 'f32');
  if (stride < n || stride % 4) throw new Error(`${label}: output stride ${stride} for ${n} columns`);

  const quadsPerRow = n / 4;
  const invocations = rows * quadsPerRow;
  const dispatch = grid1d(invocations);
  const outputs: Record<string, BufferSource> = {};
  if (output) outputs.outE4 = output;
  if (outputF16) outputs.outF16 = outputF16;
  if (outputF32) outputs.outF32 = outputF32;

  return kernel({
    label,
    kind: 'gemm_f16',
    workgroupSize: [64],
    dispatch: [...dispatch],
    inputs: { activations: input, weights },
    outputs,
    body: (views: Record<string, TSLNode>) => {
      // `id.x + id.y * 65535 * 64` of the reference, from workgroup and local ids (R12).
      const linear = workgroupId.x
        .add(workgroupId.y.mul(u(MAX_GROUPS)))
        .mul(u(64))
        .add(localId.x)
        .toVar();
      If(linear.lessThan(u(invocations)), () => {
        const row = linear.div(u(quadsPerRow)).toVar();
        const quad = linear.mod(u(quadsPerRow)).mul(u(4)).toVar();
        const values = [0, 1, 2, 3].map(() => f(0).toVar());
        const inputRow = row.mul(u(input.channels)).toVar();
        Loop(
          { start: u(0), end: u(k), update: 8, type: 'uint', condition: '<', name: 'base' },
          ({ base }: { base: TSLNode }) => {
            const a = Array.from({ length: 8 }, (_, i) =>
              loadHalf(views.activations, inputRow.add(base).add(u(i))).toVar(),
            );
            const weightRow = base.mul(u(weights.paddedN)).add(quad).toVar();
            for (let c = 0; c < 4; ++c) {
              const b = Array.from({ length: 8 }, (_, i) =>
                loadHalf(views.weights, weightRow.add(u(i * weights.paddedN + c))).toVar(),
              );
              values[c].assign(nrFdpaF16x8(a, b, values[c]));
            }
          },
        );
        const index = row.mul(u(stride)).add(quad).toVar();
        if (outputF32) {
          for (let c = 0; c < 4; ++c) views.outF32.element(index.add(u(c))).assign(bitsOf(values[c]));
        }
        const halves = outputF16 || output ? values.map((value) => nrF16Bits(value).toVar()) : [];
        if (outputF16) {
          const word = index.shiftRight(u(1));
          views.outF16.element(word).assign(packHalfPair(halves[0], halves[1]));
          views.outF16.element(word.add(u(1))).assign(packHalfPair(halves[2], halves[3]));
        }
        if (output) {
          const codes = halves.map((bits) => nrEncodeE4m3(bits));
          views.outE4.element(index.shiftRight(u(2))).assign(packWord4(codes[0], codes[1], codes[2], codes[3]));
        }
      });
    },
  });
}
