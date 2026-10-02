// Shifted-window attention (`attend_window_tiled`) as a three.js TSL compute kernel.
//
// Port of the composed production window attention of OpenDLSS-NR's WebGPU port (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f41/ports/browser-webgpu/src/window, entry point
// `attend_window_tiled` of `windowAttentionCode()` = base.js + tiled.js through the transforms of variants.js) to
// three.js TSL. Design Appendix A.3 lists its byte-level rules; `windowAttention.audit.md` maps every line of the
// composed WGSL to the code below.
//
// One workgroup per (head, window, block of `queries` queries): the cosine norms of the window's 64 tokens are
// computed from the raw qkv halves, K (natural key order) and V (physical, 4x4-tiled order) are published to E4M3 and
// staged, each query's 64 scores are two FDPA-16 groups seeded with the learned prior and mapped by `exp_weight`, the
// softmax denominator is the reference's fixed tree of half adds, the weights are published to E4M3, and each output
// component is four FDPA-16 groups over the keys in physical order, published as E4M3 bytes four to a word.
//
// Differences from the reference that do not change a byte (each argued in the audit):
//   * no f16 type: every half operation is the f32 operation followed by a rounding to half (exact for + - * / sqrt),
//     and `f16_bits(round_f16(x)) == f16_bits(x)` lets a publication take the unrounded product; the roundings use
//     the short exact forms of attentionNumerics.ts (`nrRoundHalf` = round_f16, `nrPublishE4Value` = publish_fp8);
//   * workgroup memory holds f32 values (24 KiB at 32 queries) with lifetimes overlapped: the norms live in the V
//     array until V is staged, the scores replace K once every score is in registers, the reciprocals replace Q;
//   * the norm's stride-4/2/1 tree is evaluated by one lane per token instead of three barriers (same adds, same
//     order); V is staged after the scores; the value products use `w * v * 2^(13-e)` instead of `(4w)(4v)2^(9-e)`
//     (both exact and equal: weights <= 1.125, |v| <= 448);
//   * thread mapping: `16 * queries` invocations (512 at 32 queries, as the reference; 256 at 16).

import {
  Fn,
  If,
  Loop,
  int,
  localId,
  max,
  sqrt,
  trunc,
  uint,
  uintBitsToFloat,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';

import type { NRKernel, WindowAttentionBuffers, WindowAttentionSpec } from '../types.js';
import { windowPhase } from '../geometry.js';
import { foldedGroupY, kernel } from '../tsl/KernelBuilder.js';
import { nrE4m3Exponent, nrExpWeight, nrF16Exponent } from '../tsl/numerics.js';
import { e4Code, halfAdd, halfFold, nrPublishE4Value, nrRoundHalf } from './attentionNumerics.js';
import { f, i, loadF32, loadHalf, packWord4, pick, u, type TSLNode } from '../tsl/packed.js';

/** A WGSL `for` loop over `0..count-1` (u32, named `name`): one copy of the body in the generated code. */
const forRange = (name: string, count: number, body: (index: TSLNode) => void): void => {
  Loop({ start: u(0), end: u(count), type: 'uint', condition: '<', name }, (params: Record<string, TSLNode>) => {
    body(params[name]);
  });
};

/** Queries per workgroup: 32 is the reference's tile; 16 halves the workgroup (tests prove both give equal bytes). */
export type WindowQueries = 16 | 32;

export interface WindowAttentionOptions {
  /** Queries per workgroup (default 32, the reference's). */
  queries?: WindowQueries;
}

/** Workgroup memory the kernel declares, in bytes. */
export const windowAttentionWorkgroupBytes = (queries: WindowQueries = 32): number =>
  // q vec4[queries * 8] + k/scores vec4[512] + v/norms vec4[512] + packed K/V u32[512] + packed Q/weights u32[queries * 16]
  queries * 8 * 16 + 512 * 16 + 512 * 16 + 512 * 4 + queries * 16 * 4;

/** Workgroup counts of one window attention, as `graph.js` `windowAttention` dispatches them. */
export function windowAttentionDispatch(
  { width, height, heads, phase }: Pick<WindowAttentionSpec, 'width' | 'height' | 'heads' | 'phase'>,
  queries: WindowQueries = 32,
): [number, number, number] {
  const [shiftX, shiftY] = windowPhase(phase);
  const tasks = Math.ceil((width + shiftX) / 8) * Math.ceil((height + shiftY) / 8) * (64 / queries);
  return [heads, Math.min(tasks, 65535), Math.ceil(tasks / 65535)];
}

/** `packed_window_exponents`: byte `e4m3_exponent + 106` per nonzero lane, zero for +-0; lane x lowest. */
/** One byte of `packed_window_exponents`. */
const exponentByte = (lane: TSLNode): TSLNode =>
  pick(lane.notEqual(f(0)), uint(nrE4m3Exponent(lane).add(i(106))), u(0));

/** `x * x`. */
const squareOf = (value: TSLNode): TSLNode => value.mul(value);

const nrPackedWindowExponents = Fn(([v]: [TSLNode]) => {
  return packWord4(exponentByte(v.x), exponentByte(v.y), exponentByte(v.z), exponentByte(v.w));
}).setLayout({ name: 'nr_packed_window_exponents', type: 'uint', inputs: [{ name: 'v', type: 'vec4' }] });

/** `maximum_window_exponent`: max of `current` and the largest byte of `packed` (rotate-and-max). */
const nrMaximumWindowExponent = Fn(([current, packed]: [TSLNode, TSLNode]) => {
  const rotated8 = packed.shiftLeft(u(8)).bitOr(packed.shiftRight(u(24)));
  const rotated16 = packed.shiftLeft(u(16)).bitOr(packed.shiftRight(u(16)));
  const rotated24 = packed.shiftLeft(u(24)).bitOr(packed.shiftRight(u(8)));
  return max(current, max(max(packed, rotated8), max(rotated16, rotated24)).shiftRight(u(24)));
}).setLayout({
  name: 'nr_maximum_window_exponent',
  type: 'uint',
  inputs: [
    { name: 'current', type: 'uint' },
    { name: 'packed', type: 'uint' },
  ],
});

/** Byte-maximum start of a group: 191 (exponent -21) for a zero accumulator, else its half exponent + 212. */
const groupStart = (accumulator: TSLNode): TSLNode =>
  pick(accumulator.notEqual(f(0)), uint(nrF16Exponent(accumulator).add(i(212))), u(191));

/** `bitcast<f32>(u32(e) << 23u)`. */
const pow2Biased = (biased: TSLNode): TSLNode => uintBitsToFloat(uint(biased).shiftLeft(u(23)));

/** Physical (4x4-tiled) token -> natural row-major token inside an 8x8 window (`inverse_tiled_token`). */
const inverseTiledToken = (token: TSLNode): TSLNode => {
  const tile = token.shiftRight(u(4));
  const within = token.bitAnd(u(15));
  const x = tile
    .bitAnd(u(1))
    .mul(u(4))
    .add(within.bitAnd(u(3)));
  const y = tile
    .shiftRight(u(1))
    .mul(u(4))
    .add(within.shiftRight(u(2)));
  return y.mul(u(8)).add(x);
};

/**
 * One shifted-window attention dispatch (`attend_window_tiled`, label `${label} attend`). Writes the E4M3 attended
 * values of every token of the `width x height` field; rows past `width * height` are left untouched.
 */
export function createWindowAttention(
  spec: WindowAttentionSpec,
  buffers: WindowAttentionBuffers,
  options: WindowAttentionOptions = {},
): NRKernel {
  const { width, height, heads, phase, label } = spec;
  const { qkv, prior, scales, attended } = buffers;
  const queries = options.queries ?? 32;
  if (queries !== 16 && queries !== 32) throw new RangeError(`${label}: queries must be 16 or 32`);
  const channels = heads * 32;
  const tokens = width * height;
  if (!(width > 0 && height > 0 && heads > 0)) throw new RangeError(`${label}: empty window attention`);
  if (qkv.format !== 'f16' || qkv.channels !== channels * 3 || qkv.rows < tokens) {
    throw new Error(`${label}: qkv must be f16 [>= ${tokens}][${channels * 3}]`);
  }
  if (attended.format !== 'e4' || attended.channels !== channels || attended.rows < tokens) {
    throw new Error(`${label}: attended must be e4 [>= ${tokens}][${channels}]`);
  }
  if (prior.count < heads * 4096) throw new Error(`${label}: prior holds ${prior.count} halves, needs ${heads * 4096}`);
  if (scales.count < heads) throw new Error(`${label}: ${scales.count} scales for ${heads} heads`);

  const [shiftX, shiftY] = windowPhase(phase);
  const windowsX = Math.ceil((width + shiftX) / 8);
  const windowsY = Math.ceil((height + shiftY) / 8);
  const blocks = 64 / queries;
  const tasks = windowsX * windowsY * blocks;
  const threads = queries * 16;
  const stride3 = channels * 3;

  return kernel({
    label: `${label} attend`,
    kind: 'window_attend',
    workgroupSize: [threads],
    dispatch: windowAttentionDispatch(spec, queries),
    inputs: { qkv, scales, prior },
    outputs: { attended },
    body: ({ qkv: qkvView, scales: scaleView, prior: priorView, attended: out }) => {
      const half = (base: TSLNode, offset: TSLNode): TSLNode => loadHalf(qkvView, base.add(offset));
      // Workgroup memory (f32 / u32; see the header for the lifetimes):
      //   q         Q [query][channel/4]; then the reciprocals in .x
      //   k         norm lanes; then K [channel/4][natural key]; then scores, then weights [query][physical/4]
      //   v         norms (.x q, .y k) by token; then V [component][physical/4]
      //   packed_k  K exponents [channel/4][key]; then V exponents [physical/4][component]
      //   packed_s  Q exponents [query][channel/4]; then weight exponents [query][physical/4]
      const sharedQ = workgroupArray('vec4', queries * 8).setName('nr_window_q');
      const sharedK = workgroupArray('vec4', 512).setName('nr_window_k');
      const sharedV = workgroupArray('vec4', 512).setName('nr_window_v');
      const packedK = workgroupArray('uint', 512).setName('nr_window_packed_k');
      const packedS = workgroupArray('uint', queries * 16).setName('nr_window_packed_s');

      // One FDPA-16 group of a score (`tiled_qk_0` / `tiled_qk_16`): Q channels 4g0..4g0+15 against K of `key`.
      const qkGroup = Fn(([query, key, g0, accumulator]: [TSLNode, TSLNode, TSLNode, TSLNode]) => {
        const maximum = groupStart(accumulator).toVar();
        for (let c = 0; c < 4; ++c) {
          const g = g0.add(u(c));
          maximum.assign(
            nrMaximumWindowExponent(
              maximum,
              packedS.element(query.mul(u(8)).add(g)).add(packedK.element(g.mul(u(64)).add(key))),
            ),
          );
        }
        const exponent = int(maximum).sub(i(212)).toVar();
        const alignment = pow2Biased(i(140).sub(exponent)).toVar();
        const sum = trunc(accumulator.mul(alignment)).toVar();
        for (let c = 0; c < 4; ++c) {
          const g = g0.add(u(c));
          const a = sharedQ.element(query.mul(u(8)).add(g)).toVar();
          const b = sharedK.element(g.mul(u(64)).add(key)).toVar();
          for (const lane of ['x', 'y', 'z', 'w']) sum.addAssign(trunc(a[lane].mul(b[lane]).mul(alignment)));
        }
        return nrRoundHalf(pick(sum.equal(f(0)), f(0), sum).mul(pow2Biased(exponent.add(i(114)))));
      }).setLayout({
        name: 'nr_window_qk_group',
        type: 'float',
        inputs: [
          { name: 'query', type: 'uint' },
          { name: 'key', type: 'uint' },
          { name: 'g0', type: 'uint' },
          { name: 'accumulator', type: 'float' },
        ],
      });

      // One FDPA-16 group of an output component (`tiled_value_*`): weights of physical keys 4g0..4g0+15 against V.
      const valueGroup = Fn(([query, component, g0, accumulator]: [TSLNode, TSLNode, TSLNode, TSLNode]) => {
        const maximum = groupStart(accumulator).toVar();
        for (let c = 0; c < 4; ++c) {
          const g = g0.add(u(c));
          maximum.assign(
            nrMaximumWindowExponent(
              maximum,
              packedS.element(query.mul(u(16)).add(g)).add(packedK.element(g.mul(u(32)).add(component))),
            ),
          );
        }
        const exponent = int(maximum).sub(i(212)).toVar();
        const alignment = pow2Biased(i(140).sub(exponent)).toVar();
        const sum = trunc(accumulator.mul(alignment)).toVar();
        for (let c = 0; c < 4; ++c) {
          const g = g0.add(u(c));
          const w = sharedK.element(query.mul(u(16)).add(g)).toVar();
          const v = sharedV.element(component.mul(u(16)).add(g)).toVar();
          for (const lane of ['x', 'y', 'z', 'w']) sum.addAssign(trunc(w[lane].mul(v[lane]).mul(alignment)));
        }
        return nrRoundHalf(pick(sum.equal(f(0)), f(0), sum).mul(pow2Biased(exponent.add(i(114)))));
      }).setLayout({
        name: 'nr_window_value_group',
        type: 'float',
        inputs: [
          { name: 'query', type: 'uint' },
          { name: 'component', type: 'uint' },
          { name: 'g0', type: 'uint' },
          { name: 'accumulator', type: 'float' },
        ],
      });

      const lane = localId.x;
      const task = foldedGroupY().toVar();
      const head = workgroupId.x;

      If(task.lessThan(u(tasks)), () => {
        const window = task.div(u(blocks));
        const queryOrigin = task.mod(u(blocks)).mul(u(queries)).toVar();
        const wx = int(window.mod(u(windowsX)).mul(u(8)))
          .sub(i(shiftX))
          .toVar();
        const wy = int(window.div(u(windowsX)).mul(u(8)))
          .sub(i(shiftY))
          .toVar();
        const headBase = head.mul(u(96)).toVar();

        // Field position of natural window token `token` (uint): [inField, field row (0 outside), half index of the
        // row's head slice].
        const locate = (token: TSLNode): [TSLNode, TSLNode, TSLNode] => {
          const x = wx.add(int(token.mod(u(8))));
          const y = wy.add(int(token.div(u(8))));
          const inField = x
            .greaterThanEqual(i(0))
            .and(y.greaterThanEqual(i(0)))
            .and(x.lessThan(i(width)))
            .and(y.lessThan(i(height)))
            .toVar();
          const row = pick(inField, uint(y).mul(u(width)).add(uint(x)), u(0));
          return [inField, row, row.mul(u(stride3)).add(headBase).toVar()];
        };

        // 1. Cosine norms. Lane (token, component) folds its four squares exactly as nr_norm_fma does
        //    (f16(q0 * q0 + f16(q16 * q16)), products exact) and adds the two halves; one lane per token then runs the
        //    stride-4/2/1 tree of half adds, ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7)), and publishes f16(1 / sqrt(sum)).
        for (let first = 0; first < 512; first += threads) {
          const item = lane.add(u(first)).toVar();
          const token = item.shiftRight(u(3));
          const component = item.bitAnd(u(7));
          const [inField, , base] = locate(token);
          const sums = vec4(f(0), f(0), f(0), f(0)).toVar();
          forRange('part', 2, (part) => {
            const at = (c: number): TSLNode =>
              pick(inField, half(base.add(component), part.mul(u(32)).add(u(c))), f(0));
            const low = nrRoundHalf(squareOf(at(0)).add(nrRoundHalf(squareOf(at(16)))));
            const high = nrRoundHalf(squareOf(at(8)).add(nrRoundHalf(squareOf(at(24)))));
            sums.element(part).assign(halfAdd(low, high));
          });
          sharedK.element(item).assign(sums);
        }
        workgroupBarrier();
        If(lane.lessThan(u(64)), () => {
          const norms = vec4(f(0), f(0), f(0), f(0)).toVar();
          forRange('part', 2, (part) => {
            const x = (c: TSLNode): TSLNode => sharedK.element(lane.mul(u(8)).add(c)).element(part);
            const total = f(0).toVar();
            forRange('h', 2, (h) => {
              const pair = halfAdd(halfAdd(x(h), x(h.add(u(4)))), halfAdd(x(h.add(u(2))), x(h.add(u(6)))));
              total.assign(halfFold(h.equal(u(0)), total, pair));
            });
            norms.element(part).assign(nrRoundHalf(f(1).div(sqrt(total))));
          });
          sharedV.element(lane).assign(norms);
        });
        workgroupBarrier();

        // 2. Stage K (natural key order; published k * normK) and Q (published (q * normQ) * f16(scale)).
        const scale = nrRoundHalf(loadF32(scaleView, head)).toVar();
        for (let first = 0; first < 512; first += threads) {
          const item = lane.add(u(first)).toVar();
          const key = item.bitAnd(u(63));
          const group = item.shiftRight(u(6));
          const [inField, , base] = locate(key);
          const normK = sharedV.element(key).y;
          const k = vec4(f(0), f(0), f(0), f(0)).toVar();
          forRange('j', 4, (j) => {
            const value = half(base.add(group.mul(u(4))), j.add(u(32)));
            k.element(j).assign(pick(inField, nrPublishE4Value(value.mul(normK)), f(0)));
          });
          sharedK.element(group.mul(u(64)).add(key)).assign(k);
          packedK.element(group.mul(u(64)).add(key)).assign(nrPackedWindowExponents(k));
        }
        for (let first = 0; first < queries * 8; first += threads) {
          const item = lane.add(u(first)).toVar();
          const body = (): void => {
            const query = queryOrigin.add(item.shiftRight(u(3)));
            const group = item.bitAnd(u(7));
            const [inField, , base] = locate(query);
            const normQ = sharedV.element(query).x;
            const q = vec4(f(0), f(0), f(0), f(0)).toVar();
            forRange('j', 4, (j) => {
              const value = half(base.add(group.mul(u(4))), j);
              q.element(j).assign(pick(inField, nrPublishE4Value(nrRoundHalf(value.mul(normQ)).mul(scale)), f(0)));
            });
            sharedQ.element(item).assign(q);
            packedS.element(item).assign(nrPackedWindowExponents(q));
          };
          if (queries * 8 - first >= threads) body();
          else If(item.lessThan(u(queries * 8)), body);
        }
        workgroupBarrier();

        // 3. Scores, kept in registers: lane = (query, physical key group); two FDPA groups seeded with the prior,
        //    then exp_weight. V is staged meanwhile (its array held the norms, which are no longer read).
        const scoreQuery = lane.shiftRight(u(4)).toVar();
        const scoreGroup = lane.bitAnd(u(15)).toVar();
        const naturalQuery = queryOrigin.add(scoreQuery).toVar();
        const scores = vec4(f(0), f(0), f(0), f(0)).toVar();
        forRange('j', 4, (j) => {
          const key = inverseTiledToken(scoreGroup.mul(u(4)).add(j)).toVar();
          const priorValue = loadHalf(priorView, head.mul(u(64)).add(naturalQuery).mul(u(64)).add(key));
          const low = qkGroup(scoreQuery, key, u(0), priorValue);
          scores.element(j).assign(nrExpWeight(qkGroup(scoreQuery, key, u(4), low)));
        });
        for (let first = 0; first < 512; first += threads) {
          const item = lane.add(u(first)).toVar();
          const component = item.bitAnd(u(31));
          const group = item.shiftRight(u(5));
          const v = vec4(f(0), f(0), f(0), f(0)).toVar();
          forRange('j', 4, (j) => {
            const [inField, , base] = locate(inverseTiledToken(group.mul(u(4)).add(j)));
            v.element(j).assign(pick(inField, nrPublishE4Value(half(base.add(component), u(64))), f(0)));
          });
          sharedV.element(component.mul(u(16)).add(group)).assign(v);
        }
        workgroupBarrier();
        sharedK.element(lane).assign(scores);
        workgroupBarrier();

        // 4. The denominator: tiled_softmax_pair / tiled_softmax_sum over physical keys as left folds of half adds:
        //    pair = ((b01 + b23) + b45) + b67, parity sum = ((p0 + p1) + p2) + p3, total = even + odd; f16(1 / total).
        //    Meanwhile V's packed exponents replace K's.
        If(lane.lessThan(u(queries)), () => {
          const s = (index: TSLNode): TSLNode =>
            sharedK.element(lane.mul(u(16)).add(index.shiftRight(u(2)))).element(index.bitAnd(u(3)));
          const total = f(0).toVar();
          forRange('parity', 2, (parity) => {
            const sum = f(0).toVar();
            forRange('p', 4, (p) => {
              const pair = f(0).toVar();
              forRange('m', 4, (m) => {
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
          sharedQ.element(lane).assign(vec4(nrRoundHalf(f(1).div(total)), f(0), f(0), f(0)));
        });
        for (let first = 0; first < 512; first += threads) {
          const item = lane.add(u(first)).toVar();
          const component = item.bitAnd(u(31));
          const group = item.shiftRight(u(5));
          packedK.element(item).assign(nrPackedWindowExponents(sharedV.element(component.mul(u(16)).add(group))));
        }
        workgroupBarrier();

        // 5. Weights: publish(f16(score * reciprocal)), in place, with their packed exponents.
        const reciprocal = sharedQ.element(scoreQuery).x.toVar();
        const weights = vec4(f(0), f(0), f(0), f(0)).toVar();
        forRange('j', 4, (j) => {
          weights.element(j).assign(nrPublishE4Value(scores.element(j).mul(reciprocal)));
        });
        sharedK.element(lane).assign(weights);
        packedS.element(lane).assign(nrPackedWindowExponents(weights));
        workgroupBarrier();

        // 6. Values: lane = (query, four components); four FDPA groups each over physical keys; one E4M3 word.
        for (let first = 0; first < queries * 8; first += threads) {
          const item = lane.add(u(first)).toVar();
          const body = (): void => {
            const query = item.shiftRight(u(3)).toVar();
            const firstComponent = item.bitAnd(u(7)).mul(u(4)).toVar();
            const [inField, row] = locate(queryOrigin.add(query));
            If(inField, () => {
              const word = u(0).toVar();
              forRange('c', 4, (c) => {
                const component = firstComponent.add(c);
                const value = f(0).toVar();
                forRange('group', 4, (group) => {
                  value.assign(valueGroup(query, component, group.mul(u(4)), value));
                });
                word.assign(word.bitOr(e4Code(value).shiftLeft(c.mul(u(8)))));
              });
              out
                .element(
                  row
                    .mul(u(channels))
                    .add(head.mul(u(32)))
                    .add(firstComponent)
                    .shiftRight(u(2)),
                )
                .assign(word);
            });
          };
          if (queries * 8 - first >= threads) body();
          else If(item.lessThan(u(queries * 8)), body);
        }
      });
    },
  });
}
