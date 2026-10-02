// The FP8 GEMM: every E4M3 x E4M3 matrix multiply of the network (330 of the 451 dispatches of a frame).
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Ports the semantics of the reference's *composed*
// production kernel, `variantCode({ output, residual: 'e4', batched, tile128: true })` of
// ports/browser-webgpu/src/matmul/variants.js, as the graph dispatches it (src/graph.js `Graph.gemm`). The byte-level
// rules are in the design's Appendix A.1; test/oracle/gemm.ts `oracleGemmFp8` is the same arithmetic on the CPU (checked
// against the reference on f16 hardware, test/browser/run-fp8-gemm-chrome.mjs); src/kernels/gemmFp8.audit.md maps the
// composed WGSL to this file line by line.
//
// What the reference computes, per output (row, column):
//   acc = +0, or roundF16(residual * scale) with -0 made +0             (the skip seeds the accumulator)
//   for each K step of 32, for each half (logical k 0-15, then 16-31):  (one Ada m16n8k32 FDPA reduction each)
//     E   = max(acc != 0 ? max(exp(acc), -14) : -21, max over pairs with both operands nonzero of ea + eb)
//     acc = roundF16(trunc(acc * 2^(13-E)) + sum trunc(a * b * 2^(13-E)) as 2^(E-13) units);  a zero result is +0
//   K partitions (ViT): at the end of each span, part = (first ? acc : roundF16(part + acc)); acc = 0
//   publish: E4 `publish_e4_code` (-0 -> 0x00) or the SiLU code table, and/or the half bits
// Everything after the operand decode is exact in f32: products of two E4 values have <= 8 significant bits, every
// aligned term is an integer < 2^15, a group's sum is an integer < 2^20, and the powers of two come from exponent bits,
// so the only rounding is the explicit `nrRoundF16` per group. The reference does the products in f16 on operands
// scaled by 4 (exact because model.js caps |w| <= 9); the values are the same.
//
// How this kernel computes it (thread mapping is free; the reduction order per output is not, and is kept):
//   * a workgroup owns a 32x32 output tile (the reference's tile), 128 invocations of 2 rows x 4 columns, so an E4
//     output is one whole word per row and a half output two;
//   * per K step, the tile's 32x32 A bytes and 32x32 B bytes are decoded once into workgroup memory through a 256-entry
//     table (tables.ts `e4OperandTable`; value as f32, exponent as i32 with -100 for zero / the NaN code, which no
//     pair maximum can reach), A row-major and B k-major;
//   * the two 16-k groups run as a `Loop`, each in two passes (shared exponent, then aligned sums), four k at a time
//     with vec4 operands.

import {
  Fn,
  If,
  Loop,
  floatBitsToUint,
  int,
  invocationLocalIndex,
  ivec4,
  localId,
  max,
  trunc,
  uintBitsToFloat,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';

import type { GemmFp8Buffers, GemmSpec, NRKernel, NRTensor } from '../types.js';
import { foldedGroupY, kernel, type BufferSource } from '../tsl/KernelBuilder.js';
import { nrDecodeE4m3, nrF16Bits, nrPow2, nrPublishE4CodeGemm, nrRoundF16 } from '../tsl/numerics.js';
import { f, i, loadE4, loadHalf, packHalfPair, packWord4, pick, u, type TSLNode } from '../tsl/packed.js';
import { E4_OPERAND_EXPONENT_BIAS, e4OperandTableAttribute, siluCodeTableAttribute } from './tables.js';

const MAX_GROUPS = 65535;
/** Rows (and columns) of the output tile one workgroup owns. */
const TILE = 32;

/**
 * The accumulator's part in a group's shared exponent: `max(exp(acc), -14)`, or -21 for a zero accumulator (the
 * composed `quad_fdpa_16_*` prologue). Read off the f32 bits (no `abs`, FXC R16); an infinite accumulator gives 128.
 */
export const nrGemmAccumulatorExponent = Fn(([accumulator]: [TSLNode]) => {
  const bits = floatBitsToUint(accumulator);
  const exponent = max(int(bits.shiftRight(u(23)).bitAnd(u(0xff))).sub(i(127)), i(-14));
  return pick(bits.bitAnd(u(0x7fffffff)).equal(u(0)), i(-21), exponent);
}).setLayout({ name: 'nr_gemm_accumulator_exponent', type: 'int', inputs: [{ name: 'accumulator', type: 'float' }] });

/**
 * The end of a group: `roundF16(units * 2^(E-13))`, and a zero result is +0 (`select(rounded, 0, rounded == 0)`). The
 * zero is canonicalized on the bits, so no compiler can treat it as `x`.
 */
export const nrGemmGroupFinish = Fn(([units, maximumExponent]: [TSLNode, TSLNode]) => {
  const rounded = floatBitsToUint(nrRoundF16(units.mul(nrPow2(maximumExponent.sub(i(13))))));
  return uintBitsToFloat(pick(rounded.bitAnd(u(0x7fffffff)).equal(u(0)), u(0), rounded));
}).setLayout({
  name: 'nr_gemm_group_finish',
  type: 'float',
  inputs: [
    { name: 'units', type: 'float' },
    { name: 'maximum_exponent', type: 'int' },
  ],
});

/** The decoded value of an `e4OperandTable` entry. */
const operandValue = (entry: TSLNode): TSLNode => uintBitsToFloat(entry.bitAnd(u(0xfff00000)));

/** The exponent of an `e4OperandTable` entry (-100 for a zero or NaN code). */
const operandExponent = (entry: TSLNode): TSLNode => int(entry.bitAnd(u(0xff))).sub(i(E4_OPERAND_EXPONENT_BIAS));

/** `max` over the four lanes of an `ivec4`. */
const maxLanes = (v: TSLNode): TSLNode => max(max(v.x, v.y), max(v.z, v.w));

/** Sum of the four lanes of a `vec4` (integers < 2^15: exact in any order). */
const sumLanes = (v: TSLNode): TSLNode => v.x.add(v.y).add(v.z.add(v.w));

/**
 * The part of `packedWeightIndex(k, column, n)` (the model file's MMA fragment order, `packed_weight_index`) that does
 * not depend on the K tile: `kInTile` (a multiple of 4, as a `uint` node) and `column`. The K tile adds
 * `kTile * n * 32`.
 */
function weightIndexInTile(kInTile: TSLNode, column: TSLNode): TSLNode {
  const nTile = column.shiftRight(u(7));
  const nInTile = column.bitAnd(u(127));
  const nHalf = nInTile.shiftRight(u(6));
  const nGroup = nInTile.bitAnd(u(63)).shiftRight(u(4));
  const nInGroup = nInTile.bitAnd(u(15));
  const lane = nInGroup
    .bitAnd(u(7))
    .shiftLeft(u(2))
    .bitOr(kInTile.bitAnd(u(15)).shiftRight(u(2)));
  const byteInLane = nInGroup
    .shiftRight(u(3))
    .shiftLeft(u(3))
    .bitOr(kInTile.shiftRight(u(4)).shiftLeft(u(2)));
  return nTile
    .mul(u(4096))
    .add(nHalf.mul(u(2048)))
    .add(nGroup.mul(u(512)))
    .add(lane.mul(u(16)))
    .add(byteInLane);
}

const requireTensor = (tensor: NRTensor | undefined, what: string, label: string): NRTensor => {
  if (!tensor) throw new Error(`${label}: ${what} is required`);
  return tensor;
};

/** Workgroup counts of a GEMM, as graph.js dispatches it: `[ceil(n/32) * batches, rowTiles folded past 65535]`. */
export function gemmFp8Dispatch(spec: Pick<GemmSpec, 'rows' | 'n' | 'batches'>): [number, number, number] {
  const rowTiles = Math.ceil(spec.rows / TILE);
  return [Math.ceil(spec.n / TILE) * spec.batches, Math.min(rowTiles, MAX_GROUPS), Math.ceil(rowTiles / MAX_GROUPS)];
}

/**
 * One FP8 GEMM dispatch (the reference's `Graph.gemm`): `output[row][batch * n + column]` from E4 activations
 * `input[row][...]` and an FP8 matrix in fragment order, optionally seeded from a skip tensor scaled per column, with
 * K partitions, SiLU publication, and E4 / half / dual outputs. Shapes and strides are baked into the WGSL.
 *
 * Requirements (the reference's, checked): `k % 32 == 0`, `n % 16 == 0`, `weights.k == k * batches`,
 * `weights.batchK == k`, `weights.n == n`; SiLU only with E4 output; a residual needs `scale` (n halves) and the
 * output's row stride; E4 and half outputs share one stride. Input rows are read at `input.channels` stride with
 * batch `b` at column `b * k` (or column 0 when `broadcast`); a stride that is not a multiple of 4 is rejected.
 */
export function createGemmFp8(spec: GemmSpec, buffers: GemmFp8Buffers): NRKernel {
  const { rows, k, n, batches, broadcast, partition, silu, output: mode, residual: residualSpec, label } = spec;
  const { input, weights } = buffers;
  if (!Number.isInteger(rows) || rows < 1) throw new RangeError(`${label}: rows ${rows}`);
  if (k < 32 || k % 32) throw new RangeError(`${label}: K ${k} is not a positive multiple of 32`);
  if (n < 16 || n % 16) throw new RangeError(`${label}: N ${n} is not a positive multiple of 16`);
  if (!Number.isInteger(batches) || batches < 1) throw new RangeError(`${label}: batches ${batches}`);
  if (![0, 256, 512, 1024].includes(partition)) throw new RangeError(`${label}: no K partition of ${partition}`);
  if (silu && mode !== 'e4') throw new Error(`${label} activates on a boundary that is not E4M3`);
  if (weights.k !== k * batches || weights.batchK !== k || weights.n !== n) {
    throw new Error(
      `${label} dispatches ${batches}x${k}x${n} against a ${weights.k}(${weights.batchK})x${weights.n} matrix`,
    );
  }
  const outputE4 = mode === 'half' ? undefined : requireTensor(buffers.output, 'output (E4)', label);
  const outputHalf = mode === 'e4' ? undefined : requireTensor(buffers.outputF16, 'outputF16', label);
  if (mode === 'e4' && buffers.outputF16) throw new Error(`${label}: outputF16 given to an E4-only GEMM`);
  if (mode === 'half' && buffers.output) throw new Error(`${label}: output given to a half-only GEMM`);
  const target = (outputE4 ?? outputHalf)!;
  const outputChannels = target.channels;
  if (outputE4 && outputE4.format !== 'e4') throw new Error(`${label}: output must be an e4 tensor`);
  if (outputHalf && outputHalf.format !== 'f16') throw new Error(`${label}: outputF16 must be an f16 tensor`);
  if (outputE4 && outputHalf && outputHalf.channels !== outputChannels) {
    throw new Error(`${label} publishes both tensors at one index, so they must share a stride`);
  }
  if (outputChannels < batches * n || outputChannels % 4) {
    throw new Error(`${label}: output stride ${outputChannels} cannot hold ${batches}x${n} columns in whole words`);
  }
  if (input.format !== 'e4') throw new Error(`${label}: input must be an e4 tensor`);
  const inputChannels = input.channels;
  if (inputChannels % 4 || inputChannels < (broadcast ? k : k * batches)) {
    throw new Error(`${label}: input stride ${inputChannels} for ${broadcast ? 1 : batches}x${k} channels`);
  }
  for (const tensor of [input, outputE4, outputHalf]) {
    if (tensor && tensor.rows < rows) throw new Error(`${label}: ${tensor.label} has ${tensor.rows} < ${rows} rows`);
  }
  let residual: NRTensor | undefined;
  if (residualSpec) {
    residual = requireTensor(buffers.residual, 'residual', label);
    if (residual.format !== residualSpec.format) {
      throw new Error(`${label}: residual is ${residual.format}, the spec says ${residualSpec.format}`);
    }
    if (residual.channels !== outputChannels) {
      throw new Error(`${label} reads its skip at the output index, so the strides must agree`);
    }
    if (!buffers.scale || buffers.scale.count < n) throw new Error(`${label}: a residual needs ${n} scale halves`);
  } else if (buffers.residual) {
    throw new Error(`${label}: residual given but the spec has none`);
  }

  const dispatch = gemmFp8Dispatch(spec);
  const rowTiles = Math.ceil(rows / TILE);
  const columnGroups = Math.ceil(n / TILE);
  const folded = dispatch[2] > 1;
  const partialRows = rows % TILE !== 0 || (folded && rowTiles % MAX_GROUPS !== 0);
  const partialColumns = n % TILE !== 0;
  const kTiles = k / TILE;
  const span = partition || 1024;

  const inputs: Record<string, BufferSource> = { input, weights, operands: { attribute: e4OperandTableAttribute() } };
  if (residual) inputs.residual = residual;
  if (residual) inputs.scale = buffers.scale!;
  if (silu) inputs.silu = { attribute: siluCodeTableAttribute() };
  const outputs: Record<string, BufferSource> = {};
  if (outputE4) outputs.output = outputE4;
  if (outputHalf) outputs.outputHalf = outputHalf;

  return kernel({
    label,
    kind: 'gemm_fp8',
    workgroupSize: [8, 16, 1],
    dispatch,
    inputs,
    outputs,
    body: (views: Record<string, TSLNode>) => {
      const aValues = workgroupArray('vec4', TILE * 8).setName('nr_tile_a');
      const aExponents = workgroupArray('ivec4', TILE * 8).setName('nr_tile_ea');
      const bValues = workgroupArray('vec4', TILE * 8).setName('nr_tile_b');
      const bExponents = workgroupArray('ivec4', TILE * 8).setName('nr_tile_eb');

      const rowGroup = (folded ? foldedGroupY() : workgroupId.y).toVar();
      const batch = batches > 1 ? workgroupId.x.div(u(columnGroups)).toVar() : null;
      const columnGroup = batches > 1 ? workgroupId.x.mod(u(columnGroups)).toVar() : workgroupId.x;
      const rowBase = rowGroup
        .mul(u(TILE))
        .add(localId.y.mul(u(2)))
        .toVar();
      const columnBase = columnGroup
        .mul(u(TILE))
        .add(localId.x.mul(u(4)))
        .toVar();
      // Byte / value index of output (rowBase + r, columnBase) in the output (and skip) tensors.
      const outputIndex = (r: number): TSLNode => {
        let index = rowBase.add(u(r)).mul(u(outputChannels)).add(columnBase);
        if (batch) index = index.add(batch.mul(u(n)));
        return index;
      };
      const rowValid = (r: number): TSLNode => rowBase.add(u(r)).lessThan(u(rows));
      const columnsValid = partialColumns ? columnBase.lessThan(u(n)) : null;
      const outputValid = (r: number): TSLNode | null => {
        const conditions = [partialRows ? rowValid(r) : null, columnsValid].filter((c) => c !== null);
        return conditions.length === 0
          ? null
          : conditions.length === 1
            ? conditions[0]
            : conditions[0].and(conditions[1]);
      };

      // Accumulators of the 2 x 4 outputs, seeded from the skip: roundF16(0.0 + residual * scale), -0 made +0.
      const accumulators: TSLNode[] = [];
      for (let r = 0; r < 2; ++r) {
        const index = residual ? outputIndex(r).toVar() : null;
        for (let c = 0; c < 4; ++c) {
          if (!residual) {
            accumulators.push(f(0).toVar());
            continue;
          }
          const at = index!.add(u(c));
          const skip =
            residual.format === 'e4' ? nrDecodeE4m3(loadE4(views.residual, at)) : loadHalf(views.residual, at);
          const scaled = skip.mul(loadHalf(views.scale, columnBase.add(u(c))));
          const seedBits = floatBitsToUint(nrRoundF16(scaled));
          const seed = uintBitsToFloat(pick(seedBits.bitAnd(u(0x7fffffff)).equal(u(0)), u(0), seedBits));
          const valid = outputValid(r);
          accumulators.push((valid ? pick(valid, seed, f(0)) : seed).toVar());
        }
      }
      const partitions = partition ? accumulators.map(() => f(0).toVar()) : null;

      // Loop-invariant parts of the tile loads. Each invocation loads tile words t = local index + 128 j, j = 0, 1:
      // A word t is row t / 8, logical k 4 (t % 8) .. +3; B word t is column t / 8, logical k 4 (t % 8) .. +3.
      const tileWords = [0, 1].map((j) => {
        const t = invocationLocalIndex.add(u(128 * j)).toVar();
        const q = t.bitAnd(u(7));
        const aRow = rowGroup.mul(u(TILE)).add(t.shiftRight(u(3)));
        // native_chained_input_index(4q): logical k pairs (4q, 4q+1) and (4q+2, 4q+3) sit at physical bytes
        // 16 (q / 4) + 2 (q % 4) and that + 8 of the row's K tile.
        let aByte = aRow
          .mul(u(inputChannels))
          .add(q.bitAnd(u(4)).mul(u(4)))
          .add(q.bitAnd(u(3)).mul(u(2)));
        if (batch && !broadcast) aByte = aByte.add(batch.mul(u(k)));
        const bColumn = columnGroup.mul(u(TILE)).add(t.shiftRight(u(3)));
        let bByte = weightIndexInTile(q.mul(u(4)), bColumn);
        if (batch) bByte = bByte.add(batch.mul(u(k * n)));
        return {
          t,
          bSlot: q
            .mul(u(TILE))
            .add(t.shiftRight(u(3)))
            .toVar(),
          aByte: aByte.toVar(),
          aValid: aRow.lessThan(u(rows)).toVar(),
          bByte: bByte.toVar(),
          bValid: partialColumns ? bColumn.lessThan(u(n)).toVar() : null,
        };
      });

      const exponents = accumulators.map(() => i(0).toVar());
      const scales = accumulators.map(() => f(1).toVar());
      const sums = accumulators.map(() => f(0).toVar());
      const row0 = localId.y.mul(u(2 * 8)).toVar(); // A slot of (row 2 ly, q = 0); row 2 ly + 1 is 8 further
      const column0 = localId.x.mul(u(4)).toVar(); // B slot of (q = 0, column 4 lx)

      Loop(
        { start: u(0), end: u(kTiles), type: 'uint', condition: '<', name: 'kTile' },
        ({ kTile }: { kTile: TSLNode }) => {
          // Decode this K step's operands into workgroup memory.
          for (const word of tileWords) {
            const aByte = word.aByte.add(kTile.mul(u(TILE))).toVar();
            const shift = aByte.bitAnd(u(3)).mul(u(8)).toVar();
            const low = pick(word.aValid, views.input.element(aByte.shiftRight(u(2))), u(0));
            const high = pick(word.aValid, views.input.element(aByte.add(u(8)).shiftRight(u(2))), u(0));
            const aCodes = [
              low.shiftRight(shift).bitAnd(u(0xff)),
              low.shiftRight(shift.add(u(8))).bitAnd(u(0xff)),
              high.shiftRight(shift).bitAnd(u(0xff)),
              high.shiftRight(shift.add(u(8))).bitAnd(u(0xff)),
            ].map((code) => views.operands.element(code).toVar());
            aValues.element(word.t).assign(vec4(...aCodes.map(operandValue)));
            aExponents.element(word.t).assign(ivec4(...aCodes.map(operandExponent)));

            const bAddress = word.bByte.add(kTile.mul(u(n * TILE))).shiftRight(u(2));
            const bWord = (
              word.bValid ? pick(word.bValid, views.weights.element(bAddress), u(0)) : views.weights.element(bAddress)
            ).toVar();
            const bCodes = [0, 1, 2, 3].map((lane) =>
              views.operands.element(bWord.shiftRight(u(lane * 8)).bitAnd(u(0xff))).toVar(),
            );
            bValues.element(word.bSlot).assign(vec4(...bCodes.map(operandValue)));
            bExponents.element(word.bSlot).assign(ivec4(...bCodes.map(operandExponent)));
          }
          workgroupBarrier();

          // The two 16-product groups (`quad_fdpa_16_0`, `quad_fdpa_16_16`).
          Loop(
            { start: u(0), end: u(2), type: 'uint', condition: '<', name: 'half' },
            ({ half }: { half: TSLNode }) => {
              for (let o = 0; o < 8; ++o) exponents[o].assign(nrGemmAccumulatorExponent(accumulators[o]));
              const quads = (body: (q: TSLNode) => void) =>
                Loop(
                  { start: u(0), end: u(4), type: 'uint', condition: '<', name: 'quad' },
                  ({ quad }: { quad: TSLNode }) => body(half.mul(u(4)).add(quad).toVar()),
                );
              // Pass 1: the shared exponent over the accumulator and every pair with both operands nonzero.
              quads((q) => {
                const a = [0, 1].map((r) => aExponents.element(row0.add(u(r * 8)).add(q)).toVar());
                const b = [0, 1, 2, 3].map((c) => bExponents.element(q.mul(u(TILE)).add(column0).add(u(c))).toVar());
                for (let r = 0; r < 2; ++r) {
                  for (let c = 0; c < 4; ++c)
                    exponents[r * 4 + c].assign(max(exponents[r * 4 + c], maxLanes(a[r].add(b[c]))));
                }
              });
              for (let o = 0; o < 8; ++o) {
                scales[o].assign(nrPow2(i(13).sub(exponents[o])));
                sums[o].assign(trunc(accumulators[o].mul(scales[o])));
              }
              // Pass 2: the aligned, truncated terms, summed exactly.
              quads((q) => {
                const a = [0, 1].map((r) => aValues.element(row0.add(u(r * 8)).add(q)).toVar());
                const b = [0, 1, 2, 3].map((c) => bValues.element(q.mul(u(TILE)).add(column0).add(u(c))).toVar());
                for (let r = 0; r < 2; ++r) {
                  for (let c = 0; c < 4; ++c) {
                    const o = r * 4 + c;
                    sums[o].addAssign(sumLanes(trunc(a[r].mul(b[c]).mul(scales[o]))));
                  }
                }
              });
              for (let o = 0; o < 8; ++o) accumulators[o].assign(nrGemmGroupFinish(sums[o], exponents[o]));
            },
          );

          if (partitions) {
            // A K partition ends: its sum is published to half and added to the earlier ones.
            const end = kTile.add(u(1));
            If(
              end
                .mul(u(TILE))
                .mod(u(span))
                .equal(u(0))
                .or(end.equal(u(kTiles))),
              () => {
                const first = kTile.mul(u(TILE)).lessThan(u(span));
                for (let o = 0; o < 8; ++o) {
                  partitions[o].assign(pick(first, accumulators[o], nrRoundF16(partitions[o].add(accumulators[o]))));
                  accumulators[o].assign(f(0));
                }
              },
            );
          }
          workgroupBarrier();
        },
      );

      const results = partitions ?? accumulators;
      for (let r = 0; r < 2; ++r) {
        const index = outputIndex(r).toVar();
        const write = () => {
          const values = results.slice(r * 4, r * 4 + 4);
          if (outputE4) {
            const codes = values.map((value) =>
              silu ? loadE4(views.silu, nrF16Bits(value)) : nrPublishE4CodeGemm(value),
            );
            views.output.element(index.shiftRight(u(2))).assign(packWord4(codes[0], codes[1], codes[2], codes[3]));
          }
          if (outputHalf) {
            const halves = values.map((value) => nrF16Bits(value).toVar());
            const word = index.shiftRight(u(1));
            views.outputHalf.element(word).assign(packHalfPair(halves[0], halves[1]));
            views.outputHalf.element(word.add(u(1))).assign(packHalfPair(halves[2], halves[3]));
          }
        };
        const valid = outputValid(r);
        if (valid) If(valid, write);
        else write();
      }
    },
  });
}
