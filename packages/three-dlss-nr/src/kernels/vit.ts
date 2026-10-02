// The global ViT attention (`vit_normalize`, `vit_attend`) as three.js TSL compute kernels.
//
// Port of OpenDLSS-NR's ports/browser-webgpu/shaders/vit.wgsl (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/shaders/vit.wgsl) to three.js TSL.
// vit.wgsl is already f32 + bit-pattern code over numerics.wgsl, so this is a literal port (design Appendix A.4):
// the same thread mapping, loops, reductions and roundings. `vit.audit.md` maps it line by line. Differences that do
// not change a byte:
//   * shapes are JS constants (`tokens`, `heads`, `paddedTokens` = the module override `PADDED_TOKENS`);
//   * `round_f16(sqrt(32.0))` is the constant 5.65625 (Tint folds `sqrt(32.0)` to the f32 5.656854; its half is
//     5.65625) and `vit_exp_weight(0.0)` stays a runtime call;
//   * fixed-count reductions the reference unrolls into straight code (`tree_sum16`, `softmax_pair`) are written as
//     WGSL loops computing the same left folds in the same order, which keeps FXC (D3D12) compile times in seconds.
//
// The attention differs from the window attention in four ways (see vit.wgsl): an extra sqrt(32) on the query, pair
// squares summed in f32 and rounded once, its own exponential, and unnormalized weights whose reciprocal is applied to
// the value sum. Padding tokens (rows `tokens..paddedTokens` of the normalized tensor) must be zero: their scores are
// exp(0) and are subtracted from the denominator once.

import {
  Fn,
  If,
  Loop,
  floatBitsToUint,
  localId,
  sqrt,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';

import { grid1d } from '../geometry.js';
import type { NRKernel, VitAttendBuffers, VitNormalizeBuffers, VitSpec } from '../types.js';
import { kernel } from '../tsl/KernelBuilder.js';
import {
  nrDecodeE4m3,
  nrF13Cover,
  nrF13Finish,
  nrF13Start,
  nrF13Term,
  nrPow2,
  nrVitExpWeight,
} from '../tsl/numerics.js';
import { e4Code, halfAdd, halfFold, nrPublishE4Value, nrRoundHalf } from './attentionNumerics.js';
import { f, i, loadE4, loadF32, loadHalf, packWord4, pick, u, type TSLNode } from '../tsl/packed.js';

/** `round_f16(sqrt(32.0))`: the half nearest the f32 sqrt(32) = 5.6568541526794434. */
const HEAD_SCALE = 5.65625;

/** A WGSL `for` loop over `start..end-1` in steps of `step` (u32, named `name`). */
const forRange = (name: string, start: TSLNode, end: number, step: number, body: (index: TSLNode) => void): void => {
  Loop({ start, end: u(end), type: 'uint', condition: '<', name, update: step }, (params: Record<string, TSLNode>) => {
    body(params[name]);
  });
};

/**
 * `ada_fp8_fdpa16` exactly as numerics.wgsl writes it - two loops over 16 operand pairs (`a(i)`, `b(i)` build the
 * operands from the loop index), the f13 helpers, and an infinite or NaN accumulator passed through - so each operand
 * load appears once in the generated code.
 */
function fdpa16(a: (index: TSLNode) => TSLNode, b: (index: TSLNode) => TSLNode, accumulator: TSLNode): TSLNode {
  const maximum = nrF13Start(accumulator).toVar();
  forRange('cover', u(0), 16, 1, (index) => {
    maximum.assign(nrF13Cover(maximum, a(index), b(index)));
  });
  const scale = nrPow2(i(13).sub(maximum)).toVar();
  const units = nrF13Term(accumulator, scale).toVar();
  forRange('term', u(0), 16, 1, (index) => {
    units.addAssign(nrF13Term(a(index).mul(b(index)), scale));
  });
  const finite = floatBitsToUint(accumulator).bitAnd(u(0x7f800000)).notEqual(u(0x7f800000));
  return pick(finite, nrF13Finish(units, maximum), accumulator);
}

function checkSpec({ tokens, heads, paddedTokens, label }: VitSpec): void {
  if (!(tokens > 0 && heads > 0)) throw new RangeError(`${label}: empty ViT`);
  if (paddedTokens !== tokens + 63 - ((tokens + 63) % 64)) {
    throw new RangeError(`${label}: paddedTokens ${paddedTokens} is not ${tokens} rounded up to 64`);
  }
}

/**
 * `vit_normalize`: cosine-normalize q (times sqrt(32) and the learned per-head scale) and k per token and head, and
 * publish q, k and v to E4M3. Writes rows `0..tokens-1` of `normalized`; the padding rows are never written.
 * Label `${label} normalize`; one invocation per (token, head, four channels).
 */
export function createVitNormalize(spec: VitSpec, buffers: VitNormalizeBuffers): NRKernel {
  checkSpec(spec);
  const { tokens, heads, paddedTokens, label } = spec;
  const { qkv, scales, normalized } = buffers;
  const channels = heads * 32;
  if (qkv.format !== 'f16' || qkv.channels !== channels * 3 || qkv.rows < tokens) {
    throw new Error(`${label}: qkv must be f16 [>= ${tokens}][${channels * 3}]`);
  }
  if (normalized.format !== 'e4' || normalized.channels !== channels * 3 || normalized.rows < paddedTokens) {
    throw new Error(`${label}: normalized must be e4 [>= ${paddedTokens}][${channels * 3}]`);
  }
  if (scales.count < heads) throw new Error(`${label}: ${scales.count} scales for ${heads} heads`);
  const count = tokens * heads * 8;
  const perToken = heads * 8;

  return kernel({
    label: `${label} normalize`,
    kind: 'vit_normalize',
    workgroupSize: [64],
    dispatch: grid1d(count),
    inputs: { qkv, scales },
    outputs: { normalized },
    body: ({ qkv: qkvView, scales: scaleView, normalized: out }) => {
      // Per-invocation scratch for the 16 pair sums of `vit_norm` (each invocation reads only its own 16).
      const pairSums = workgroupArray('float', 64 * 16).setName('nr_vit_pair_sums');
      const scratch = localId.x.mul(u(16)).toVar();
      const r = (c: TSLNode): TSLNode => pairSums.element(scratch.add(c));
      const s8 = (c: TSLNode): TSLNode => nrRoundHalf(r(c).add(r(c.add(u(8)))));
      // `vit_norm` (with `tree_sum16`) over the 32 halves from half index `first`:
      //   r[c] = f16(v[c]^2 + f16(v[c+16]^2)), s8[c] = f16(r[c] + r[c+8]), s4[c] = f16(s8[c] + s8[c+4]),
      //   total = f16(f16(s4[0] + s4[2]) + f16(s4[1] + s4[3])), norm = f16(1 / sqrt(total)),
      // written as loops (h over the two outer pairs, m over the two s4 of a pair) so each rounding appears once.
      const vitNorm = (first: TSLNode): TSLNode => {
        forRange('c', u(0), 16, 1, (c) => {
          const low = loadHalf(qkvView, first.add(c));
          const high = loadHalf(qkvView, first.add(c).add(u(16)));
          pairSums.element(scratch.add(c)).assign(nrRoundHalf(low.mul(low).add(nrRoundHalf(high.mul(high)))));
        });
        const total = f(0).toVar();
        forRange('h', u(0), 2, 1, (h) => {
          const outer = f(0).toVar();
          forRange('m', u(0), 2, 1, (m) => {
            const c = h.add(m.mul(u(2)));
            const s4 = nrRoundHalf(s8(c).add(s8(c.add(u(4)))));
            outer.assign(pick(m.equal(u(0)), s4, nrRoundHalf(outer.add(s4))));
          });
          total.assign(pick(h.equal(u(0)), outer, nrRoundHalf(total.add(outer))));
        });
        return nrRoundHalf(f(1).div(sqrt(total)));
      };

      const linear = workgroupId.x
        .add(workgroupId.y.mul(u(65535)))
        .mul(u(64))
        .add(localId.x)
        .toVar();
      If(linear.lessThan(u(count)), () => {
        const token = linear.div(u(perToken));
        const head = linear.mod(u(perToken)).div(u(8)).toVar();
        const quad = linear.mod(u(8)).mul(u(4)).toVar();
        const base = token
          .mul(u(channels * 3))
          .add(head.mul(u(96)))
          .toVar();
        // q's norm (part 0) and k's (part 1), one loop body.
        const norms = vec4(f(0), f(0), f(0), f(0)).toVar();
        forRange('part', u(0), 2, 1, (part) => {
          norms.element(part).assign(vitNorm(base.add(part.mul(u(32)))));
        });
        const qNorm = norms.x;
        const kNorm = norms.y;
        const learned = nrRoundHalf(loadF32(scaleView, head)).toVar();
        const wordQ = u(0).toVar();
        const wordK = u(0).toVar();
        const wordV = u(0).toVar();
        forRange('i', u(0), 4, 1, (index) => {
          const c = base.add(quad).add(index);
          const shift = index.mul(u(8));
          // Three half multiplies, in this order: the norm, then sqrt(head_dim), then the learned scale.
          const nq = nrRoundHalf(
            nrRoundHalf(nrRoundHalf(loadHalf(qkvView, c).mul(qNorm)).mul(f(HEAD_SCALE))).mul(learned),
          );
          wordQ.assign(wordQ.bitOr(e4Code(nq).shiftLeft(shift)));
          wordK.assign(wordK.bitOr(e4Code(nrRoundHalf(loadHalf(qkvView, c.add(u(32))).mul(kNorm))).shiftLeft(shift)));
          wordV.assign(wordV.bitOr(e4Code(loadHalf(qkvView, c.add(u(64)))).shiftLeft(shift)));
        });
        out.element(base.add(quad).shiftRight(u(2))).assign(wordQ);
        out.element(base.add(u(32)).add(quad).shiftRight(u(2))).assign(wordK);
        out.element(base.add(u(64)).add(quad).shiftRight(u(2))).assign(wordV);
      });
    },
  });
}

/**
 * `vit_attend`: one workgroup of 64 per (head, query token): scores against every padded key, the blocked softmax
 * denominator with the padding correction, unnormalized E4M3 weights, the value sum over all padded keys, and the
 * E4M3 output `out = f16(value * f16(1 / total))`. Label `${label} attend`.
 */
export function createVitAttend(spec: VitSpec, buffers: VitAttendBuffers): NRKernel {
  checkSpec(spec);
  const { tokens, heads, paddedTokens, label } = spec;
  const { normalized, attended } = buffers;
  const channels = heads * 32;
  const stride3 = channels * 3;
  if (normalized.format !== 'e4' || normalized.channels !== stride3 || normalized.rows < paddedTokens) {
    throw new Error(`${label}: normalized must be e4 [>= ${paddedTokens}][${stride3}]`);
  }
  if (attended.format !== 'e4' || attended.channels !== channels || attended.rows < tokens) {
    throw new Error(`${label}: attended must be e4 [>= ${tokens}][${channels}]`);
  }
  if (tokens > 65535) throw new RangeError(`${label}: ${tokens} tokens exceed one dispatch row`);
  const padding = paddedTokens - tokens;

  return kernel({
    label: `${label} attend`,
    kind: 'vit_attend',
    workgroupSize: [64],
    dispatch: [heads, tokens, 1],
    inputs: { normalized },
    outputs: { attended },
    body: ({ normalized: input, attended: out }) => {
      const scores = workgroupArray('float', paddedTokens).setName('nr_vit_scores');
      const weightValues = workgroupArray('float', paddedTokens).setName('nr_vit_weights');
      const queryVector = workgroupArray('float', 32).setName('nr_vit_query');
      const reciprocal = workgroupArray('float', 1).setName('nr_vit_reciprocal');

      const thread = localId.x;
      const head = workgroupId.x;
      const token = workgroupId.y;
      const headBase = head.mul(u(96)).toVar();
      const normalizedAt = (index: TSLNode): TSLNode => nrDecodeE4m3(loadE4(input, index));

      // The score of one key: two FDPA-16 steps of the query against the key's 32 channels, then vit_exp_weight.
      const keyScore = Fn(([key, base]: [TSLNode, TSLNode]) => {
        const keyBase = key.mul(u(stride3)).add(base).add(u(32)).toVar();
        const score = f(0).toVar();
        forRange('step', u(0), 32, 16, (c0) => {
          score.assign(
            fdpa16(
              (index) => queryVector.element(c0.add(index)),
              (index) => normalizedAt(keyBase.add(c0).add(index)),
              score,
            ),
          );
        });
        return nrVitExpWeight(score);
      }).setLayout({
        name: 'nr_vit_key_score',
        type: 'float',
        inputs: [
          { name: 'key', type: 'uint' },
          { name: 'base', type: 'uint' },
        ],
      });

      // softmax_pair / softmax64 over the 64 scores from `base`, as left folds of half adds:
      //   pair = f16(f16(f16(a + b) + c) + d) with a..d = f16(s[k + 16m] + s[k + 16m + 8]),
      //   parity total = f16(f16(f16(p0 + p1) + p2) + p3), block = f16(even + odd).
      const softmax64 = Fn(([base]: [TSLNode]) => {
        const s = (index: TSLNode): TSLNode => scores.element(base.add(index));
        const total = f(0).toVar();
        forRange('parity', u(0), 2, 1, (parity) => {
          const sum = f(0).toVar();
          forRange('p', u(0), 4, 1, (p) => {
            const pair = f(0).toVar();
            forRange('m', u(0), 4, 1, (m) => {
              const key = p
                .mul(u(2))
                .add(parity)
                .add(m.mul(u(16)));
              pair.assign(halfFold(m.equal(u(0)), pair, halfAdd(s(key), s(key.add(u(8))))));
            });
            sum.assign(halfFold(p.equal(u(0)), sum, pair));
          });
          total.assign(halfFold(parity.equal(u(0)), total, sum));
        });
        return total;
      }).setLayout({ name: 'nr_vit_softmax64', type: 'float', inputs: [{ name: 'base', type: 'uint' }] });

      If(thread.lessThan(u(32)), () => {
        queryVector.element(thread).assign(normalizedAt(token.mul(u(stride3)).add(headBase).add(thread)));
      });
      workgroupBarrier();

      forRange('key', thread, paddedTokens, 64, (key) => {
        scores.element(key).assign(keyScore(key, headBase));
      });
      workgroupBarrier();

      If(thread.equal(u(0)), () => {
        const total = f(0).toVar();
        forRange('block', u(0), paddedTokens, 64, (block) => {
          total.assign(halfAdd(total, softmax64(block)));
        });
        if (padding > 0) {
          const correction = nrRoundHalf(nrVitExpWeight(f(0)).mul(f(padding)));
          total.assign(nrRoundHalf(total.sub(correction)));
        }
        reciprocal.element(u(0)).assign(nrRoundHalf(f(1).div(total)));
      });
      // The weights are published unnormalized; the reciprocal reaches the result through the value sum.
      forRange('key', thread, paddedTokens, 64, (key) => {
        weightValues.element(key).assign(nrPublishE4Value(scores.element(key)));
      });
      workgroupBarrier();

      // Each of the first 32 threads takes one component and walks every key in k16 steps.
      If(thread.lessThan(u(32)), () => {
        const value = f(0).toVar();
        const component = headBase.add(u(64)).add(thread).toVar();
        forRange('k0', u(0), paddedTokens, 16, (k0) => {
          value.assign(
            fdpa16(
              (index) => weightValues.element(k0.add(index)),
              (index) => normalizedAt(k0.add(index).mul(u(stride3)).add(component)),
              value,
            ),
          );
        });
        scores.element(thread).assign(nrRoundHalf(value.mul(reciprocal.element(u(0))))); // reuse: scores are finished
      });
      workgroupBarrier();

      If(thread.lessThan(u(8)), () => {
        const code = (lane: number): TSLNode => e4Code(scores.element(thread.mul(u(4)).add(u(lane))));
        out
          .element(
            token
              .mul(u(channels))
              .add(head.mul(u(32)))
              .shiftRight(u(2))
              .add(thread),
          )
          .assign(packWord4(code(0), code(1), code(2), code(3)));
      });
    },
  });
}
