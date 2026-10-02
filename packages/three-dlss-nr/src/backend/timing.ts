// Frame timing shared by every NR backend, so the TSL port and the reference shim are measured the same way.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41).
//
// The two backends submit their frames differently (the reference encodes one command buffer of its own; three's
// `renderer.compute` encodes and submits inside the renderer), so neither can put timestamp writes on the frame's
// own passes without knowing the other's internals. Instead the timer brackets the frame on the queue: an empty
// compute pass that writes a timestamp at its beginning is submitted before the frame, and another that writes one at
// its end after it. The difference is the GPU time of everything the queue ran in between: the frame. The same
// bracket also yields the wall time from the first submit to `onSubmittedWorkDone`, the fallback when the device has
// no `timestamp-query`.
//
// Notes: browsers quantize timestamps (Chrome to 100 us unless started with
// `--enable-dawn-features=allow_unsafe_apis` or `--enable-webgpu-developer-features`), which does not matter at
// frame scale. Work another part of the app submits between the brackets is counted too: time one thing at a time.

import { BUFFER_USAGE, MAP_MODE_READ } from './gpuFlags.js';
import type { NRFrameTiming } from './NRBackend.js';

/** Brackets submissions with timestamp queries (when available and asked for) and wall time. */
export class NRFrameTimer {
  readonly device: GPUDevice;
  /** `timestamp-query` is enabled on the device. */
  readonly timestamps: boolean;
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private readBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.timestamps = device.features.has('timestamp-query');
  }

  /**
   * Time `submit`, which must submit the frame's work to `device.queue` and return without waiting for the GPU (a
   * submit that waits still gives a right wall time, but its GPU time then includes the idle wait). Resolves after
   * the GPU has finished the frame.
   */
  async measure(submit: () => unknown, { gpu = true }: { gpu?: boolean } = {}): Promise<NRFrameTiming> {
    const stamp = gpu && this.timestamps;
    if (stamp) this.submitStamp('begin');
    const started = performance.now();
    await submit();
    if (stamp) this.submitStamp('end');
    await this.device.queue.onSubmittedWorkDone();
    const wallMilliseconds = performance.now() - started;
    if (!stamp) return { method: 'submitted-work-done', gpuMilliseconds: null, wallMilliseconds };
    const read = this.readBuffer!;
    await read.mapAsync(MAP_MODE_READ);
    const [begin, end] = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    const gpuMilliseconds = end > begin && begin > 0n ? Number(end - begin) / 1e6 : null;
    return { method: 'timestamp-query', gpuMilliseconds, wallMilliseconds };
  }

  private submitStamp(which: 'begin' | 'end'): void {
    if (!this.querySet) {
      this.querySet = this.device.createQuerySet({ label: 'nr frame timer', type: 'timestamp', count: 2 });
      this.resolveBuffer = this.device.createBuffer({
        label: 'nr frame timer resolve',
        size: 16,
        usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
      });
      this.readBuffer = this.device.createBuffer({
        label: 'nr frame timer read',
        size: 16,
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
    }
    const encoder = this.device.createCommandEncoder({ label: `nr frame timer ${which}` });
    const pass = encoder.beginComputePass({
      label: `nr frame timer ${which}`,
      timestampWrites:
        which === 'begin'
          ? { querySet: this.querySet, beginningOfPassWriteIndex: 0 }
          : { querySet: this.querySet, endOfPassWriteIndex: 1 },
    });
    pass.end();
    if (which === 'end') {
      encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer!, 0);
      encoder.copyBufferToBuffer(this.resolveBuffer!, 0, this.readBuffer!, 0, 16);
    }
    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.readBuffer?.destroy();
    this.querySet = this.resolveBuffer = this.readBuffer = null;
  }
}

/** Minimum and median of a list of numbers (`null` entries skipped); `null` when nothing is left. */
export function summarizeMilliseconds(values: readonly (number | null)[]): { min: number; median: number } | null {
  // A fresh array from filter(), sorted in place (lib ES2022 has no toSorted).
  // oxlint-disable-next-line unicorn/no-array-sort
  const sorted = values.filter((value): value is number => value !== null).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return { min: sorted[0], median };
}
