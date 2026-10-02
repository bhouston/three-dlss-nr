// The graph's view of the attention kernels (window attention and the global ViT attention).
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). The graph (`Graph.ts`, a port of the reference's
// ports/browser-webgpu/src/graph.js) calls the attention kernels only through this module, against the spec and buffer
// types of `types.ts`, so the kernels' own modules can change without touching the graph.
//
// STUB: the attention kernels (`kernels/windowAttention.ts`, `kernels/vit.ts`) are not merged yet. Until they are,
// these factories return kernels with the reference's label, kind and dispatch size (so the graph's pass list can be
// checked against the reference), whose compute body throws when three builds it: a network that contains one cannot
// be compiled or run. Replace each body with the real factory call when the kernels land.

import { grid1d, windowPhase } from '../geometry.js';
import { kernel } from '../tsl/KernelBuilder.js';
import type {
  NRKernel,
  VitAttendBuffers,
  VitNormalizeBuffers,
  VitSpec,
  WindowAttentionBuffers,
  WindowAttentionSpec,
} from '../types.js';

/** False while the attention kernels are stubs (networks then build, but cannot compile or run). */
export const ATTENTION_KERNELS_AVAILABLE = false;

const unavailable = (label: string) => () => {
  throw new Error(`${label}: the attention kernels are not merged yet (src/graph/attention.ts is a stub)`);
};

/** Workgroup counts of one window attention, as `graph.js` `windowAttention` dispatches them (32 queries a group). */
function windowDispatch({ width, height, heads, phase }: WindowAttentionSpec): [number, number, number] {
  const [shiftX, shiftY] = windowPhase(phase);
  const tasks = Math.ceil((width + shiftX) / 8) * Math.ceil((height + shiftY) / 8) * 2;
  return [heads, Math.min(tasks, 65535), Math.ceil(tasks / 65535)];
}

/** One shifted-window attention (`attend_window_tiled`); label `${spec.label} attend`. */
export function createWindowAttentionKernel(spec: WindowAttentionSpec, buffers: WindowAttentionBuffers): NRKernel {
  const label = `${spec.label} attend`;
  return kernel({
    label,
    kind: 'window_attend',
    workgroupSize: [512],
    dispatch: windowDispatch(spec),
    inputs: { qkv: buffers.qkv, scales: buffers.scales, prior: buffers.prior },
    outputs: { attended: buffers.attended },
    body: unavailable(label),
  });
}

/** `vit_normalize`; label `${spec.label} normalize`. */
export function createVitNormalizeKernel(spec: VitSpec, buffers: VitNormalizeBuffers): NRKernel {
  const label = `${spec.label} normalize`;
  return kernel({
    label,
    kind: 'vit_normalize',
    workgroupSize: [64],
    dispatch: grid1d(spec.tokens * spec.heads * 8),
    inputs: { qkv: buffers.qkv, scales: buffers.scales },
    outputs: { normalized: buffers.normalized },
    body: unavailable(label),
  });
}

/** `vit_attend`; label `${spec.label} attend`. */
export function createVitAttendKernel(spec: VitSpec, buffers: VitAttendBuffers): NRKernel {
  const label = `${spec.label} attend`;
  return kernel({
    label,
    kind: 'vit_attend',
    workgroupSize: [64],
    dispatch: [spec.heads, spec.tokens, 1],
    inputs: { normalized: buffers.normalized },
    outputs: { attended: buffers.attended },
    body: unavailable(label),
  });
}
