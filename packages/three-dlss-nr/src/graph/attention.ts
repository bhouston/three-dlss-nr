// The graph's view of the attention kernels (window attention and the global ViT attention).
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The graph (`Graph.ts`, a port of the reference's
// ports/browser-webgpu/src/graph.js) calls the attention kernels only through this module, against the spec and buffer
// types of `types.ts`. The kernels themselves are `kernels/windowAttention.ts` and `kernels/vit.ts`; their factories
// take the block label and append ` attend` / ` normalize`, as the reference's graph does.

import { createVitAttend, createVitNormalize } from '../kernels/vit.js';
import { createWindowAttention, type WindowQueries } from '../kernels/windowAttention.js';
import type {
  NRKernel,
  VitAttendBuffers,
  VitNormalizeBuffers,
  VitSpec,
  WindowAttentionBuffers,
  WindowAttentionSpec,
} from '../types.js';

export type { WindowQueries };

/**
 * Queries per window-attention workgroup a device can run: 32 (the reference's tile, 512 invocations) when its limits
 * allow, else 16 (256 invocations). Both publish identical bytes; only the dispatch size differs.
 */
export function windowQueriesFor(device: Pick<GPUDevice, 'limits'> | undefined): WindowQueries {
  if (!device) return 32;
  const { maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = device.limits;
  return maxComputeInvocationsPerWorkgroup >= 512 && maxComputeWorkgroupSizeX >= 512 ? 32 : 16;
}

/** One shifted-window attention (`attend_window_tiled`); label `${spec.label} attend`. */
export const createWindowAttentionKernel = (
  spec: WindowAttentionSpec,
  buffers: WindowAttentionBuffers,
  queries: WindowQueries = 32,
): NRKernel => createWindowAttention(spec, buffers, { queries });

/** `vit_normalize`; label `${spec.label} normalize`. `normalized` has `paddedTokens` rows whose padding stays zero. */
export const createVitNormalizeKernel = (spec: VitSpec, buffers: VitNormalizeBuffers): NRKernel =>
  createVitNormalize(spec, buffers);

/** `vit_attend`; label `${spec.label} attend`. */
export const createVitAttendKernel = (spec: VitSpec, buffers: VitAttendBuffers): NRKernel =>
  createVitAttend(spec, buffers);
