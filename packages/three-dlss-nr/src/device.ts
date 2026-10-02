// The WebGPU device and three.js renderer the network runs on.
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). Mirrors the reference's
// ports/browser-webgpu/src/gpu.js `requestDevice`: prefer the discrete adapter and raise every limit the network
// cares about to the adapter's maximum (the only way to bind tensors over 128 MiB or use more than 16 KiB of
// workgroup storage).
//
// We create the device ourselves and hand it to `WebGPURenderer({ device })` rather than letting three request one:
// three's backend asks for a `featureLevel: 'compatibility'` adapter with default limits, and a device we own can be
// shared with the reference implementation in parity tests.
//
// `shader-f16` is requested when available but never required: the port computes in f32 and integers. The
// reference's FP8 GEMM and window attention do need it (their modules begin `enable f16;`).

import { WebGPURenderer } from 'three/webgpu';

/** Workgroup storage the port's kernels need (window attention with 32 queries keeps ~19 KiB). */
export const NR_MIN_WORKGROUP_STORAGE = 24576;

/** Limits raised to the adapter maximum. */
export const NR_RAISED_LIMITS = [
  'maxComputeWorkgroupStorageSize',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxStorageBuffersPerShaderStage',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
] as const;

/** Features requested when the adapter has them. */
export const NR_OPTIONAL_FEATURES = ['core-features-and-limits', 'timestamp-query', 'shader-f16'] as const;

export interface NRDeviceOptions {
  /** The WebGPU entry point; defaults to `navigator.gpu`. */
  gpu?: GPU;
  /** Defaults to `'high-performance'`. */
  powerPreference?: GPUPowerPreference;
  /** Extra limits (override the raised ones). */
  requiredLimits?: Record<string, number>;
  /** Log uncaptured device errors to the console (default true). */
  logErrors?: boolean;
}

export interface NRDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** `shader-f16` is enabled on the device (needed only to run the reference implementation on it). */
  shaderF16: boolean;
  /** `timestamp-query` is enabled. */
  timestampQuery: boolean;
}

/** Request an adapter and a device configured for the network. Throws if the adapter cannot run it. */
export async function createNRDevice(options: NRDeviceOptions = {}): Promise<NRDevice> {
  const gpu = options.gpu ?? (typeof navigator !== 'undefined' ? navigator.gpu : undefined);
  if (!gpu) throw new Error('WebGPU is unavailable (navigator.gpu is undefined)');
  const adapter = await gpu.requestAdapter({ powerPreference: options.powerPreference ?? 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter; on a laptop, check that the discrete GPU is selected');

  const adapterLimits = adapter.limits as unknown as Record<string, number>;
  const requiredLimits: Record<string, number> = {};
  for (const name of NR_RAISED_LIMITS) requiredLimits[name] = adapterLimits[name];
  Object.assign(requiredLimits, options.requiredLimits);
  if (requiredLimits.maxComputeWorkgroupStorageSize < NR_MIN_WORKGROUP_STORAGE) {
    throw new Error(
      `the network needs ${NR_MIN_WORKGROUP_STORAGE} bytes of workgroup storage; this adapter offers ` +
        `${requiredLimits.maxComputeWorkgroupStorageSize}`,
    );
  }
  const requiredFeatures = NR_OPTIONAL_FEATURES.filter((feature) => adapter.features.has(feature)) as GPUFeatureName[];
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  if (options.logErrors ?? true) {
    // A validation failure does not throw: the offending call becomes a no-op, which reads as "the kernel computed
    // zeros". Property form, so a renderer that sets its own handler replaces rather than stacks it.
    device.onuncapturederror = (event: GPUUncapturedErrorEvent) => {
      console.error(`WebGPU ${event.error.constructor.name}: ${event.error.message}`);
    };
  }
  void device.lost.then((info) => {
    if (info.reason !== 'destroyed') console.error(`WebGPU device lost: ${info.reason} ${info.message}`);
  });
  return {
    adapter,
    device,
    shaderF16: device.features.has('shader-f16'),
    timestampQuery: device.features.has('timestamp-query'),
  };
}

/** Problems that keep a device from running the network (empty when it can). */
export function nrDeviceProblems(device: GPUDevice): string[] {
  const problems: string[] = [];
  if (device.limits.maxComputeWorkgroupStorageSize < NR_MIN_WORKGROUP_STORAGE) {
    problems.push(
      `maxComputeWorkgroupStorageSize is ${device.limits.maxComputeWorkgroupStorageSize}; ` +
        `the network needs ${NR_MIN_WORKGROUP_STORAGE}`,
    );
  }
  if (device.limits.maxComputeInvocationsPerWorkgroup < 256) {
    problems.push(`maxComputeInvocationsPerWorkgroup is ${device.limits.maxComputeInvocationsPerWorkgroup}; need 256`);
  }
  if (device.limits.maxStorageBuffersPerShaderStage < 8) {
    problems.push(`maxStorageBuffersPerShaderStage is ${device.limits.maxStorageBuffersPerShaderStage}; need 8`);
  }
  return problems;
}

/** Throw unless `device` (e.g. an existing renderer's `renderer.backend.device`) can run the network. */
export function checkNRDeviceLimits(device: GPUDevice): void {
  const problems = nrDeviceProblems(device);
  if (problems.length) throw new Error(`this WebGPU device cannot run the network: ${problems.join('; ')}`);
}

export interface NRRendererOptions extends NRDeviceOptions {
  /** Use this device instead of requesting one. */
  device?: GPUDevice;
  /** Canvas for the renderer (headless environments pass their own). */
  canvas?: unknown;
}

/**
 * A `WebGPURenderer` on a device configured for the network, initialized. Returns the renderer and its device.
 */
export async function createNRRenderer(options: NRRendererOptions = {}): Promise<{ renderer: any; device: GPUDevice }> {
  const device = options.device ?? (await createNRDevice(options)).device;
  checkNRDeviceLimits(device);
  const renderer = new WebGPURenderer({
    device,
    antialias: false,
    ...(options.canvas !== undefined ? { canvas: options.canvas } : {}),
  });
  await renderer.init();
  if (!renderer.backend?.device) throw new Error('WebGPURenderer fell back to WebGL; the network needs WebGPU');
  return { renderer, device };
}
