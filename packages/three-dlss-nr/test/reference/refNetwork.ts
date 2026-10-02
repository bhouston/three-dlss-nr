// The whole reference network (OpenDLSS-NR's WebGPU port, by maan, MIT) in Node, on our device.
//
// Part of three-dlss-nr. Wraps reference/OpenDLSS-NR/ports/browser-webgpu/src/network.js `Network.create` with the
// two things a shared test device needs (design 5.2): always pass both `device` and `model` (otherwise
// `Network.destroy()` destroys the device), and pass `shaderBase` as a `file:` URL that the fetch shim serves.
// Needs `shader-f16` on the device (the FP8 GEMM and window attention modules enable it).

import { Model } from '@ref/model.js';
import { Network } from '@ref/network.js';
import { REFERENCE_BOUNDARIES } from '@ref/parity.js';

import { referenceShaderBase, referenceSupportsF16 } from './refKernels.js';

export { REFERENCE_BOUNDARIES };

/**
 * Load a reference `Model` from a directory URL: `synthetic://<name>` (registered with the fetch shim's
 * `registerSyntheticFiles`) or a `file:` URL of a real model directory (`NR_WEIGHTS`).
 */
export async function loadReferenceModel(device: GPUDevice, directory: string): Promise<any> {
  return new Model(device).load(directory.replace(/\/+$/, ''));
}

export interface ReferenceNetworkOptions {
  device: GPUDevice;
  /** A loaded reference `Model` (see `loadReferenceModel`). */
  model: any;
  /** Valid image size. */
  width: number;
  height: number;
  captureBoundaries?: boolean;
  /**
   * Called after the graph is recorded and before it is compiled, with the reference `Network` (e.g. truncate
   * `net.recorder.passes.length = cut` to stop after dispatch `cut`, for bisection).
   */
  after?: (network: any) => void;
  onProgress?: (message: string) => void;
}

/** A thin typed handle on the reference `Network`. */
export interface ReferenceNetwork {
  readonly network: any;
  /** Upload the input features (f32 `[fullRows][16]`). */
  writeFeatures(features: Float32Array): void;
  /** One frame; resolves when the GPU has finished. Throws if the first frame's command buffer is rejected. */
  run(): Promise<void>;
  /** The f32 RGBA head `[fullRows][4]`. */
  readHead(): Promise<Float32Array>;
  /** A captured block boundary by name (`block-N`, `transition-a-b`, `pooled-a-b`), its valid bytes. */
  readBoundary(name: string): Promise<Uint8Array>;
  /** Any tensor by the label the graph allocated it under (first match), its valid bytes. */
  readTensorByLabel(label: string): Promise<Uint8Array>;
  readonly boundaryNames: string[];
  /** Dispatch labels in order. */
  readonly dispatchLabels: string[];
  destroy(): void;
}

/** Build the reference network on a shared device and a borrowed model. */
export async function createReferenceNetwork(options: ReferenceNetworkOptions): Promise<ReferenceNetwork> {
  if (!referenceSupportsF16(options.device)) {
    throw new Error('the reference network needs shader-f16 (its FP8 GEMM and window attention enable f16)');
  }
  const network = await Network.create({
    device: options.device,
    model: options.model,
    width: options.width,
    height: options.height,
    captureBoundaries: options.captureBoundaries ?? false,
    shaderBase: referenceShaderBase(),
    after: options.after,
    onProgress: options.onProgress,
  });
  return {
    network,
    writeFeatures: (features) => network.writeFeatures(features),
    run: () => network.run(),
    readHead: () => network.readHead(),
    readBoundary: (name) => network.readBoundary(name),
    readTensorByLabel: async (label) => (await network.readTensorByLabel(label)).bytes,
    get boundaryNames() {
      return network.boundaryNames;
    },
    get dispatchLabels() {
      return network.recorder.passes
        .filter((pass: { kind: string }) => pass.kind === 'dispatch')
        .map((pass: { label: string }) => pass.label);
    },
    destroy: () => network.destroy(),
  };
}
