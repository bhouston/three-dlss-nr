// The network: 71 blocks over a six-level encoder/decoder with a global ViT at the bottom, built once into an ordered
// list of TSL compute kernels.
//
// Port of OpenDLSS-NR ports/browser-webgpu/src/graph.js (MIT, (c) 2026 maan,
// https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f41/ports/browser-webgpu/src/graph.js) to three.js TSL.
// docs/network.md in the reference explains the shape of the network and why it is that shape.
//
// The dispatch order, labels, kinds and dispatch sizes are those of the reference's `Graph.record`
// (graph.js:354-580), one for one, so a mismatch can be bisected by dispatch index on both sides (451 dispatches at
// every size). So are the tensor labels and shapes (`NRTensors`, keyed as the reference's `Tensors`), the window
// phase counters, and the boundary names. What differs is the plumbing:
//   * each dispatch is an `NRKernel` built by a factory of `kernels/` (shapes, strides and flags baked into the WGSL)
//     instead of a pipeline plus a uniform block;
//   * weights are per-matrix attributes from `NRModel` (the GEMM's skip scales, which the reference reads at a byte
//     offset into the stage buffer, `auxOffset`, are an `auxVector` of the same bytes);
//   * a boundary capture is a word-copy kernel inside the one compute pass (`createCopyWords`), where the reference
//     records a `copyBufferToBuffer` that splits its pass. Captures are not dispatches: they never shift an index.

import { fusedLayout, postFusedLayout, preFusedLayout, upsampleFusedLayout, WindowPhases } from '../geometry.js';
import type { FusedLayout, NRGeometry } from '../geometry.js';
import { createGemmF16 } from '../kernels/gemmF16.js';
import { createGemmFp8 } from '../kernels/gemmFp8.js';
import {
  createConvertF32ToF16,
  createCopyWords,
  createDownsample,
  createPostBlend,
  createUpsampleResidual,
} from '../kernels/ops.js';
import type { NRModel, NRModelTensor } from '../model/Model.js';
import { NRTensors } from '../tensors.js';
import type { FP8Matrix, HalfVector, NRKernel, NRTensor, StorageBufferAttribute } from '../types.js';
import {
  createVitAttendKernel,
  createVitNormalizeKernel,
  createWindowAttentionKernel,
  type WindowQueries,
} from './attention.js';

/** One entry of the graph's ordered kernel list. */
export interface NRGraphPass {
  readonly kernel: NRKernel;
  /**
   * Index of this dispatch among the network's dispatches (0..450), the index the reference's recorder gives it; for
   * a capture, the index of the dispatch it captures (it runs right after it).
   */
  readonly index: number;
  /** Set on boundary captures: the boundary name (`block-N`, `transition-a-b`, `pooled-a-b`). */
  readonly capture?: string;
}

export interface NRGraphOptions {
  /** Keep a copy of every block boundary (79 of them), as the reference's parity harness does. */
  captureBoundaries?: boolean;
  /**
   * Queries per window-attention workgroup: 32 (default, the reference's dispatch) or 16 for devices without 512
   * invocations per workgroup (`windowQueriesFor(device)`); identical bytes either way.
   */
  windowQueries?: WindowQueries;
}

/** The two activation tensors of a stage that the blocks ping-pong between, and the per-stage temporaries. */
interface Temporaries {
  ffn: NRTensor;
  ffnNarrow: NRTensor | null;
  ffnResidual: NRTensor;
  ffnQuantized: NRTensor;
  qkv: NRTensor;
  attended: NRTensor;
}

interface SplitTemporaries {
  branch: NRTensor;
  middle: NRTensor;
  layer0: NRTensor;
  ffnResidual: NRTensor;
  qkv: NRTensor;
  attended: NRTensor;
}

interface GemmCall {
  input: NRTensor;
  weights: FP8Matrix;
  output?: NRTensor | null;
  outputF16?: NRTensor | null;
  rows: number;
  k: number;
  n: number;
  batches?: number;
  broadcast?: boolean;
  partition?: 0 | 256 | 512 | 1024;
  silu?: boolean;
  residual?: NRTensor | null;
  /** Per-output-column skip scales (`residual` only). */
  scale?: HalfVector;
  label: string;
}

/**
 * The recorded network for one geometry: the ordered kernels of a frame, every tensor, the captured boundaries.
 * Building it allocates the CPU side of every buffer; three creates the GPU buffers when it first builds the kernels.
 */
export class NRGraph {
  readonly geometry: NRGeometry;
  readonly model: NRModel;
  readonly tensors = new NRTensors();
  /** Every kernel in submission order: the 451 dispatches, each followed by its capture when capturing. */
  readonly passes: NRGraphPass[] = [];
  /** The network's dispatches only (no captures), in order. */
  readonly dispatches: NRKernel[] = [];
  /** Captured boundary tensors by name, in capture order. */
  readonly boundaries = new Map<string, NRTensor>();
  /** The f32 input features `[fullRows][16]` (`input features`). */
  readonly features: NRTensor;
  /** The f32 RGBA head `[fullRows][4]` (`head`). */
  head!: NRTensor;
  /** Every weight attribute the kernels bind (for disposal). */
  readonly weightAttributes = new Set<StorageBufferAttribute>();

  private readonly captureBoundaries: boolean;
  private readonly windowQueries: WindowQueries;
  private readonly phases = new WindowPhases();

  constructor(model: NRModel, geometry: NRGeometry, options: NRGraphOptions = {}) {
    if (model.blockCount !== 71) {
      throw new Error(`the model has ${model.blockCount} blocks; this graph is the 71-block network`);
    }
    this.model = model;
    this.geometry = geometry;
    this.captureBoundaries = options.captureBoundaries ?? false;
    this.windowQueries = options.windowQueries ?? 32;
    this.features = this.tensors.allocate('input features', geometry.fullRows, 16, 'f32');
    this.record();
  }

  /** Dispatch labels in order. */
  get labels(): string[] {
    return this.dispatches.map((k) => k.label);
  }

  // -------------------------------------------------------------------------------------------------------------
  // Plumbing.
  // -------------------------------------------------------------------------------------------------------------

  private add(kernel: NRKernel): void {
    this.passes.push({ kernel, index: this.dispatches.length });
    this.dispatches.push(kernel);
  }

  private tensorFor(label: string, rows: number, channels: number, format: 'e4' | 'f16' | 'f32'): NRTensor {
    return this.tensors.allocate(label, rows, channels, format);
  }

  private weights<T extends { attribute: StorageBufferAttribute }>(source: T): T {
    this.weightAttributes.add(source.attribute);
    return source;
  }

  private fp8(tensor: NRModelTensor, byteOffset: number, k: number, n: number, batchK = 0): FP8Matrix {
    return this.weights(this.model.fp8Matrix(tensor, byteOffset, k, n, { batchK }));
  }

  /** The skip scales the reference's GEMM reads at `auxOffset(tensor, byteOffset, count)`. */
  private aux(tensor: NRModelTensor, byteOffset: number, count: number): HalfVector {
    return this.weights(this.model.auxVector(tensor, byteOffset, count));
  }

  private capture(name: string, source: NRTensor): void {
    if (!this.captureBoundaries) return;
    const copy = this.tensorFor(`boundary ${name}`, source.rows, source.channels, source.format);
    const kernel = createCopyWords({ label: `capture ${name}` }, { source, target: copy });
    this.passes.push({ kernel, index: this.dispatches.length - 1, capture: name });
    this.boundaries.set(name, copy);
  }

  // -------------------------------------------------------------------------------------------------------------
  // The kernels, as the graph wants to call them (graph.js:59-145).
  // -------------------------------------------------------------------------------------------------------------

  /** One FP8 matrix multiply; `residual` seeds the accumulator, scaled per column by `scale` (`Graph.gemm`). */
  private gemm(call: GemmCall): void {
    const { input, weights, rows, k, n, label } = call;
    const output = call.output ?? undefined;
    const outputF16 = call.outputF16 ?? undefined;
    const residual = call.residual ?? undefined;
    const mode = output && outputF16 ? 'dual' : output ? 'e4' : 'half';
    if (residual && residual.format === 'f32') throw new Error(`${label}: an f32 skip`);
    this.add(
      createGemmFp8(
        {
          rows,
          k,
          n,
          batches: call.batches ?? 1,
          broadcast: call.broadcast ?? false,
          partition: call.partition ?? 0,
          silu: call.silu ?? false,
          output: mode,
          residual: residual ? { format: residual.format as 'e4' | 'f16' } : null,
          label,
        },
        { input, weights, output, outputF16, residual, scale: residual ? call.scale : undefined },
      ),
    );
  }

  /** One shifted-window attention (`Graph.windowAttention`); the dispatch label is `${label} attend`. */
  private windowAttention(
    tensor: NRModelTensor,
    relative: number,
    scale: number,
    { qkv, attended }: { qkv: NRTensor; attended: NRTensor },
    width: number,
    height: number,
    heads: number,
    phase: number,
    label: string,
  ): void {
    const prior = this.weights(this.model.relativeBias(tensor, relative, heads));
    const scales = this.weights(this.model.headScales(tensor, scale, heads));
    this.add(
      createWindowAttentionKernel(
        { width, height, heads, phase: phase & 3, label },
        { qkv, prior, scales, attended },
        this.windowQueries,
      ),
    );
  }

  // -------------------------------------------------------------------------------------------------------------
  // One block (graph.js:162-326).
  // -------------------------------------------------------------------------------------------------------------

  /** FFN -> QKV -> window attention -> projection, with the two scaled skips that make it a residual block. */
  private block(args: {
    block: number;
    channels: number;
    width: number;
    height: number;
    layout: FusedLayout;
    tensor: NRModelTensor;
    temps: Temporaries;
    state: NRTensor;
    output: NRTensor | null;
    outputF16?: NRTensor | null;
    ffnSkipOverride?: NRTensor | null;
    phase: number;
  }): void {
    const { block, channels, width, height, layout, tensor, temps, state, output, phase } = args;
    const rows = width * height;
    const label = `block ${block}`;
    const residual = args.ffnSkipOverride ?? state;
    const ffnScale = this.aux(tensor, layout.ffnCosSkip, channels);

    if (layout.expertFfn) {
      const experts = layout.expertCount;
      const w2Base = layout.expand + experts * channels * 128;
      const w3Base = w2Base + experts * 128 * 32;
      this.gemm({
        input: state,
        weights: this.fp8(tensor, layout.expand, experts * channels, 128, channels),
        output: temps.ffn,
        rows,
        k: channels,
        n: 128,
        batches: experts,
        broadcast: true,
        silu: true,
        label: `${label} expert expand`,
      });
      this.gemm({
        input: temps.ffn,
        weights: this.fp8(tensor, w2Base, experts * 128, 32, 128),
        output: temps.ffnNarrow,
        rows,
        k: 128,
        n: 32,
        batches: experts,
        label: `${label} expert contract`,
      });
      this.gemm({
        input: temps.ffnNarrow!,
        weights: this.fp8(tensor, w3Base, channels, channels),
        output: temps.ffnQuantized,
        outputF16: temps.ffnResidual,
        rows,
        k: channels,
        n: channels,
        residual,
        scale: ffnScale,
        label: `${label} expert merge`,
      });
    } else {
      this.gemm({
        input: state,
        weights: this.fp8(tensor, layout.expand, channels, layout.hidden),
        output: temps.ffn,
        rows,
        k: channels,
        n: layout.hidden,
        silu: true,
        label: `${label} expand`,
      });
      this.gemm({
        input: temps.ffn,
        weights: this.fp8(tensor, layout.contractWeights, layout.hidden, channels),
        output: temps.ffnQuantized,
        outputF16: temps.ffnResidual,
        rows,
        k: layout.hidden,
        n: channels,
        residual,
        scale: ffnScale,
        label: `${label} contract`,
      });
    }

    this.gemm({
      input: temps.ffnQuantized,
      weights: this.fp8(tensor, layout.qkv, channels, channels * 3),
      outputF16: temps.qkv,
      rows,
      k: channels,
      n: channels * 3,
      label: `${label} qkv`,
    });
    this.windowAttention(tensor, layout.relative, layout.scale, temps, width, height, layout.heads, phase, label);

    // The attention skip: the E4M3 publication of the FFN for the expert blocks, the raw half for the others.
    this.gemm({
      input: temps.attended,
      weights: this.fp8(tensor, layout.projection, channels, channels),
      output,
      outputF16: args.outputF16,
      rows,
      k: channels,
      n: channels,
      residual: layout.expertFfn ? temps.ffnQuantized : temps.ffnResidual,
      scale: this.aux(tensor, layout.attnCosSkip, channels),
      label: `${label} projection`,
    });
  }

  /** The 512 stage: eight 64-wide branches widened to 256 and back, on top of one 512 -> 512 layer. */
  private splitBlock(args: {
    block: number;
    width: number;
    height: number;
    temps: SplitTemporaries;
    state: NRTensor;
    output: NRTensor;
    outputF16?: NRTensor | null;
    phase: number;
  }): void {
    const { block, width, height, temps, state, output, phase } = args;
    const rows = width * height;
    const model = this.model;
    const label = `block ${block}`;
    const channels = 512;
    const branches = 8;
    const branchChannels = 64;
    const middleChannels = 256;
    const heads = 16;
    const branchTensor = model.tensor(block, 0);
    const contract = model.tensor(block, 1);
    const qkvTensor = model.tensor(block, 2);
    const projection = model.tensor(block, 3);
    const w2Base = branches * channels * branchChannels;
    const w3Base = w2Base + branches * branchChannels * middleChannels;
    const qkvRelative = channels * channels * 3;
    const qkvScale = qkvRelative + heads * 8192;

    this.gemm({
      input: state,
      weights: this.fp8(branchTensor, 0, channels, channels),
      output: temps.branch,
      rows,
      k: channels,
      n: channels,
      label: `${label} split layer0`,
    });
    this.gemm({
      input: temps.branch,
      weights: this.fp8(branchTensor, w2Base, branches * branchChannels, middleChannels, branchChannels),
      output: temps.middle,
      rows,
      k: branchChannels,
      n: middleChannels,
      batches: branches,
      silu: true,
      label: `${label} split expand`,
    });
    this.gemm({
      input: temps.middle,
      weights: this.fp8(branchTensor, w3Base, branches * middleChannels, branchChannels, middleChannels),
      output: temps.layer0,
      rows,
      k: middleChannels,
      n: branchChannels,
      batches: branches,
      label: `${label} split contract`,
    });
    this.gemm({
      input: temps.layer0,
      weights: this.fp8(contract, 0, channels, channels),
      output: temps.ffnResidual,
      rows,
      k: channels,
      n: channels,
      residual: state,
      scale: this.aux(contract, channels * channels, channels),
      label: `${label} split merge`,
    });
    this.gemm({
      input: temps.ffnResidual,
      weights: this.fp8(qkvTensor, 0, channels, channels * 3),
      outputF16: temps.qkv,
      rows,
      k: channels,
      n: channels * 3,
      label: `${label} qkv`,
    });
    this.windowAttention(qkvTensor, qkvRelative, qkvScale, temps, width, height, heads, phase, label);
    this.gemm({
      input: temps.attended,
      weights: this.fp8(projection, 0, channels, channels),
      output,
      outputF16: args.outputF16,
      rows,
      k: channels,
      n: channels,
      residual: temps.ffnResidual,
      scale: this.aux(projection, channels * channels, channels),
      label: `${label} projection`,
    });
  }

  /** The global ViT: eight blocks whose attention spans every token of the coarsest level. */
  private vit(state: NRTensor, tokens: number): void {
    const channels = 1024;
    const heads = 32;
    const ffnChannels = 4096;
    const paddedTokens = this.geometry.paddedVitTokens;
    const model = this.model;
    const expanded = this.tensorFor('vit expand', tokens, ffnChannels, 'e4');
    const ffnResidual = this.tensorFor('vit residual', tokens, channels, 'e4');
    const qkv = this.tensorFor('vit qkv', tokens, channels * 3, 'f16');
    // Padding rows are never written and must stay zero (the attend reads every padded key).
    const normalized = this.tensorFor('vit normalized', paddedTokens, channels * 3, 'e4');
    const attended = this.tensorFor('vit attended', tokens, channels, 'e4');

    for (let block = 31; block <= 38; ++block) {
      const expand = model.tensor(block, 0);
      const contract = model.tensor(block, 1);
      const qkvTensor = model.tensor(block, 2);
      const projection = model.tensor(block, 4);
      const label = `block ${block}`;
      this.gemm({
        input: state,
        weights: this.fp8(expand, 0, channels, ffnChannels),
        output: expanded,
        rows: tokens,
        k: channels,
        n: ffnChannels,
        silu: true,
        label: `${label} expand`,
      });
      this.gemm({
        input: expanded,
        weights: this.fp8(contract, 0, ffnChannels, channels),
        output: ffnResidual,
        rows: tokens,
        k: ffnChannels,
        n: channels,
        partition: 1024,
        residual: state,
        scale: this.aux(contract, ffnChannels * channels, channels),
        label: `${label} contract`,
      });
      // The ViT's qkv tensor puts its per-head scales before the weights, where every other block puts them after.
      this.gemm({
        input: ffnResidual,
        weights: this.fp8(qkvTensor, heads * 4, channels, channels * 3),
        outputF16: qkv,
        rows: tokens,
        k: channels,
        n: channels * 3,
        partition: 512,
        label: `${label} qkv`,
      });
      const spec = { tokens, heads, paddedTokens, label };
      const scales = this.weights(model.headScales(qkvTensor, 0, heads));
      this.add(createVitNormalizeKernel(spec, { qkv, scales, normalized }));
      this.add(createVitAttendKernel(spec, { normalized, attended }));
      this.gemm({
        input: attended,
        weights: this.fp8(projection, 0, channels, channels),
        output: state,
        rows: tokens,
        k: channels,
        n: channels,
        partition: 256,
        residual: ffnResidual,
        scale: this.aux(projection, channels * channels, channels),
        label: `${label} projection`,
      });
      this.capture(`block-${block}`, state);
    }
  }

  // -------------------------------------------------------------------------------------------------------------

  private temporaries(label: string, rows: number, channels: number, layout: FusedLayout): Temporaries {
    const hidden = layout.expertFfn ? layout.expertCount * 128 : layout.hidden;
    return {
      ffn: this.tensorFor(`${label} ffn`, rows, hidden, 'e4'),
      ffnNarrow: layout.expertFfn ? this.tensorFor(`${label} ffn narrow`, rows, channels, 'e4') : null,
      ffnResidual: this.tensorFor(`${label} ffn residual`, rows, channels, 'f16'),
      ffnQuantized: this.tensorFor(`${label} ffn quantized`, rows, channels, 'e4'),
      qkv: this.tensorFor(`${label} qkv`, rows, channels * 3, 'f16'),
      attended: this.tensorFor(`${label} attended`, rows, channels, 'e4'),
    };
  }

  private splitTemporaries(label: string, rows: number): SplitTemporaries {
    return {
      branch: this.tensorFor(`${label} branch`, rows, 512, 'e4'),
      middle: this.tensorFor(`${label} middle`, rows, 2048, 'e4'),
      layer0: this.tensorFor(`${label} layer0`, rows, 512, 'e4'),
      ffnResidual: this.tensorFor(`${label} split residual`, rows, 512, 'e4'),
      qkv: this.tensorFor(`${label} split qkv`, rows, 1536, 'f16'),
      attended: this.tensorFor(`${label} split attended`, rows, 512, 'e4'),
    };
  }

  /** Record the whole network (graph.js:354-580). */
  private record(): void {
    this.phases.reset();
    const g = this.geometry;
    const model = this.model;
    const fullRows = g.fullRows;
    const [d0, d1, d2, d3, d4, d5] = g.levels;

    // ---- Block 0 at full resolution, behind the f16 input adapter.
    const preTensor = model.tensor(0);
    const preLayout = preFusedLayout();
    if (preTensor.byteLength !== preLayout.endWithoutPadding + 16) throw new Error('unexpected block 0 layout');
    const featuresHalf = this.tensorFor('features f16', fullRows, 16, 'f16');
    this.add(
      createConvertF32ToF16(
        { count: fullRows * 16, label: 'features to half' },
        { input: this.features, output: featuresHalf },
      ),
    );
    const adapterF16 = this.tensorFor('adapter f16', fullRows, 32, 'f16');
    const adapterE4 = this.tensorFor('adapter e4', fullRows, 32, 'e4');
    this.add(
      createGemmF16(
        { rows: fullRows, k: 16, n: 32, label: 'input adapter' },
        {
          input: featuresHalf,
          weights: this.weights(model.f16Matrix(preTensor, preLayout.inputAdapter!, 16, 32)),
          output: adapterE4,
          outputF16: adapterF16,
        },
      ),
    );
    const block0 = this.tensorFor('block 0 out', fullRows, 32, 'e4');
    const block0Raw = this.tensorFor('block 0 raw', fullRows, 32, 'f16');
    const fullTemps = this.temporaries('full', fullRows, 32, preLayout);
    this.block({
      block: 0,
      channels: 32,
      width: g.fullWidth,
      height: g.fullHeight,
      layout: preLayout,
      tensor: preTensor,
      temps: fullTemps,
      state: adapterE4,
      output: block0,
      outputF16: block0Raw,
      ffnSkipOverride: adapterF16,
      phase: this.phases.take(6),
    });
    this.capture('block-0', block0);

    // ---- Down to level 0 and the four 32-channel blocks there.
    const rows0 = d0.rows;
    let state = this.tensorFor('level0 in', rows0, 32, 'e4');
    this.add(
      createDownsample(
        {
          channels: 32,
          inWidth: g.fullWidth,
          inHeight: g.fullHeight,
          outWidth: d0.width,
          outHeight: d0.height,
          count: rows0 * 32,
          label: 'pool 0',
        },
        { input: block0Raw, output: state },
      ),
    );
    this.capture('transition-0-1', state);

    let scratch = this.tensorFor('level0 state', rows0, 32, 'e4');
    const level0Raw = this.tensorFor('level0 raw', rows0, 32, 'f16');
    const level0Temps = this.temporaries('level0', rows0, 32, fusedLayout(32));
    for (let block = 1; block <= 4; ++block) {
      this.block({
        block,
        channels: 32,
        width: d0.width,
        height: d0.height,
        layout: fusedLayout(32),
        tensor: model.tensor(block),
        temps: level0Temps,
        state,
        output: scratch,
        outputF16: block === 4 ? level0Raw : null,
        phase: this.phases.take(0),
      });
      [state, scratch] = [scratch, state];
      this.capture(`block-${block}`, state);
    }
    const skip32 = state;

    // ---- Encoder stages 64 / 128 / 256, each ending in a pool and a widening transition.
    const pooled32 = this.tensorFor('pool 4', d1.rows, 32, 'e4');
    this.add(
      createDownsample(
        {
          channels: 32,
          inWidth: d0.width,
          inHeight: d0.height,
          outWidth: d1.width,
          outHeight: d1.height,
          count: d1.rows * 32,
          label: 'pool 4',
        },
        { input: level0Raw, output: pooled32 },
      ),
    );
    this.capture('pooled-4-5', pooled32);
    let stageInput = this.tensorFor('stage 64 in', d1.rows, 64, 'e4');
    this.gemm({
      input: pooled32,
      weights: this.fp8(model.tensor(4), fusedLayout(32).endWithoutPadding, 32, 64),
      output: stageInput,
      rows: d1.rows,
      k: 32,
      n: 64,
      label: 'transition 4-5',
    });
    this.capture('transition-4-5', stageInput);

    const encoderStages = [
      { level: d1, next: d2, channels: 64, first: 5, last: 8, levelIndex: 1 },
      { level: d2, next: d3, channels: 128, first: 9, last: 14, levelIndex: 2 },
      { level: d3, next: d4, channels: 256, first: 15, last: 22, levelIndex: 3 },
    ];
    const skips: NRTensor[] = [];
    for (const stage of encoderStages) {
      const rows = stage.level.rows;
      const label = `encoder ${stage.channels}`;
      const layout = fusedLayout(stage.channels);
      let st = stageInput;
      let sc = this.tensorFor(`${label} state`, rows, stage.channels, 'e4');
      const raw = this.tensorFor(`${label} raw`, rows, stage.channels, 'f16');
      const temps = this.temporaries(label, rows, stage.channels, layout);
      for (let block = stage.first; block <= stage.last; ++block) {
        this.block({
          block,
          channels: stage.channels,
          width: stage.level.width,
          height: stage.level.height,
          layout,
          tensor: model.tensor(block),
          temps,
          state: st,
          output: sc,
          outputF16: block === stage.last ? raw : null,
          phase: this.phases.take(stage.levelIndex),
        });
        [st, sc] = [sc, st];
        this.capture(`block-${block}`, st);
      }
      skips.push(st);
      const pooled = this.tensorFor(`${label} pooled`, stage.next.rows, stage.channels, 'e4');
      this.add(
        createDownsample(
          {
            channels: stage.channels,
            inWidth: stage.level.width,
            inHeight: stage.level.height,
            outWidth: stage.next.width,
            outHeight: stage.next.height,
            count: stage.next.rows * stage.channels,
            label: `pool ${stage.last}`,
          },
          { input: raw, output: pooled },
        ),
      );
      this.capture(`pooled-${stage.last}-${stage.last + 1}`, pooled);
      const next = this.tensorFor(`${label} next`, stage.next.rows, stage.channels * 2, 'e4');
      this.gemm({
        input: pooled,
        weights: this.fp8(model.tensor(stage.last), layout.endWithoutPadding, stage.channels, stage.channels * 2),
        output: next,
        rows: stage.next.rows,
        k: stage.channels,
        n: stage.channels * 2,
        label: `transition ${stage.last}`,
      });
      this.capture(`transition-${stage.last}-${stage.last + 1}`, next);
      stageInput = next;
    }
    const [skip64, skip128, skip256] = skips;

    // ---- The 512 stage, the pool into the ViT, and the ViT.
    {
      const rows = d4.rows;
      let st = stageInput;
      let sc = this.tensorFor('encoder 512 state', rows, 512, 'e4');
      const raw = this.tensorFor('encoder 512 raw', rows, 512, 'f16');
      const temps = this.splitTemporaries('encoder 512', rows);
      for (let block = 23; block <= 30; ++block) {
        this.splitBlock({
          block,
          width: d4.width,
          height: d4.height,
          temps,
          state: st,
          output: sc,
          outputF16: block === 30 ? raw : null,
          phase: this.phases.take(4),
        });
        [st, sc] = [sc, st];
        this.capture(`block-${block}`, st);
      }
      const skip512 = st;
      const tokens = g.vitTokens;
      const pooled = this.tensorFor('vit pooled', tokens, 512, 'e4');
      this.add(
        createDownsample(
          {
            channels: 512,
            inWidth: d4.width,
            inHeight: d4.height,
            outWidth: d5.width,
            outHeight: d5.height,
            count: tokens * 512,
            label: 'pool 30',
          },
          { input: raw, output: pooled },
        ),
      );
      const vitState = this.tensorFor('vit state', tokens, 1024, 'e4');
      this.gemm({
        input: pooled,
        weights: this.fp8(model.tensor(30, 4), 0, 512, 1024),
        output: vitState,
        rows: tokens,
        k: 512,
        n: 1024,
        label: 'transition 30-31',
      });
      this.vit(vitState, tokens);

      // ---- Decoder 512: the ViT output projected, doubled, and merged onto the encoder skip.
      const projected = this.tensorFor('decoder 512 projection', tokens, 512, 'f16');
      this.gemm({
        input: vitState,
        weights: this.fp8(model.tensor(39), 0, 1024, 512),
        outputF16: projected,
        rows: tokens,
        k: 1024,
        n: 512,
        partition: 256,
        label: 'transition 38-39',
      });
      const merged = this.tensorFor('decoder 512 merge', d4.rows, 512, 'e4');
      this.add(
        createUpsampleResidual(
          {
            channels: 512,
            inWidth: d5.width,
            inHeight: d5.height,
            outWidth: d4.width,
            outHeight: d4.height,
            count: d4.rows * 512,
            label: 'block 39 merge',
          },
          { input: projected, skip: skip512, scale: this.aux(model.tensor(39), 1024 * 512, 512), output: merged },
        ),
      );
      this.capture('block-39', merged);

      let dst = merged;
      let dsc = this.tensorFor('decoder 512 state', d4.rows, 512, 'e4');
      const dtemps = this.splitTemporaries('decoder 512', d4.rows);
      for (let block = 40; block <= 47; ++block) {
        this.splitBlock({
          block,
          width: d4.width,
          height: d4.height,
          temps: dtemps,
          state: dst,
          output: dsc,
          phase: this.phases.take(4),
        });
        [dst, dsc] = [dsc, dst];
        this.capture(`block-${block}`, dst);
      }
      stageInput = dst;
    }

    // ---- Decoder stages 256 / 128 / 64 / 32.
    const decoderStages = [
      { low: d4, high: d3, channels: 256, first: 48, last: 55, levelIndex: 3, skip: skip256 },
      { low: d3, high: d2, channels: 128, first: 56, last: 61, levelIndex: 2, skip: skip128 },
      { low: d2, high: d1, channels: 64, first: 62, last: 65, levelIndex: 1, skip: skip64 },
      { low: d1, high: d0, channels: 32, first: 66, last: 69, levelIndex: 0, skip: skip32 },
    ];
    for (const stage of decoderStages) {
      const rows = stage.high.rows;
      const label = `decoder ${stage.channels}`;
      const transition = model.tensor(stage.first);
      const layout = upsampleFusedLayout(stage.channels * 2, stage.channels);
      if (transition.byteLength !== layout.endWithoutPadding + 16) {
        throw new Error(`unexpected upsample layout for block ${stage.first}`);
      }
      const projection = this.tensorFor(`${label} projection`, stage.low.rows, stage.channels, 'f16');
      this.gemm({
        input: stageInput,
        weights: this.fp8(transition, layout.upsampleWeight!, stage.channels * 2, stage.channels),
        outputF16: projection,
        rows: stage.low.rows,
        k: stage.channels * 2,
        n: stage.channels,
        label: `transition ->${stage.first}`,
      });
      const merged = this.tensorFor(`${label} merge`, rows, stage.channels, 'e4');
      const mergedRaw = stage.channels === 32 ? this.tensorFor(`${label} merge raw`, rows, 32, 'f16') : null;
      this.add(
        createUpsampleResidual(
          {
            channels: stage.channels,
            inWidth: stage.low.width,
            inHeight: stage.low.height,
            outWidth: stage.high.width,
            outHeight: stage.high.height,
            count: rows * stage.channels,
            label: `block ${stage.first} merge`,
          },
          {
            input: projection,
            skip: stage.skip,
            scale: this.aux(transition, layout.transitionScale!, stage.channels),
            output: merged,
            ...(mergedRaw ? { outputF16: mergedRaw } : {}),
          },
        ),
      );

      let st = merged;
      let sc = this.tensorFor(`${label} state`, rows, stage.channels, 'e4');
      const temps = this.temporaries(label, rows, stage.channels, layout);
      for (let block = stage.first; block <= stage.last; ++block) {
        this.block({
          block,
          channels: stage.channels,
          width: stage.high.width,
          height: stage.high.height,
          layout: block === stage.first ? layout : fusedLayout(stage.channels),
          tensor: model.tensor(block),
          temps,
          state: st,
          output: sc,
          ffnSkipOverride: block === stage.first ? mergedRaw : null,
          phase: this.phases.take(stage.levelIndex),
        });
        [st, sc] = [sc, st];
        this.capture(`block-${block}`, st);
      }
      stageInput = st;
    }

    // ---- The post block at full resolution, and the head.
    {
      const tensor = model.tensor(70);
      const layout = postFusedLayout();
      if (tensor.byteLength !== layout.endWithoutPadding) throw new Error('unexpected block 70 layout');
      const mergedRaw = this.tensorFor('post merge raw', fullRows, 32, 'f16');
      const merged = this.tensorFor('post merge', fullRows, 32, 'e4');
      this.add(
        createPostBlend(
          {
            channels: 32,
            inWidth: d0.width,
            inHeight: d0.height,
            outWidth: g.fullWidth,
            outHeight: g.fullHeight,
            count: fullRows * 32,
            inputScaleOffset: 0,
            skipScaleOffset: 32,
            label: 'post blend',
          },
          {
            input: stageInput,
            skip: block0,
            scales: this.weights(model.auxPair(tensor, layout.inputScale!, layout.adapterScale!, 32)),
            output: merged,
            outputF16: mergedRaw,
          },
        ),
      );
      const blockRaw = this.tensorFor('post block raw', fullRows, 32, 'f16');
      const postTemps = this.temporaries('post', fullRows, 32, layout);
      this.block({
        block: 70,
        channels: 32,
        width: g.fullWidth,
        height: g.fullHeight,
        layout,
        tensor,
        temps: postTemps,
        state: merged,
        output: null,
        outputF16: blockRaw,
        ffnSkipOverride: mergedRaw,
        phase: this.phases.take(6),
      });
      this.head = this.tensorFor('head', fullRows, 4, 'f32');
      this.add(
        createGemmF16(
          { rows: fullRows, k: 32, n: 4, label: 'head' },
          {
            input: blockRaw,
            weights: this.weights(model.f16Matrix(tensor, layout.postWeights!, 32, 4)),
            outputF32: this.head,
          },
        ),
      );
    }
  }
}
