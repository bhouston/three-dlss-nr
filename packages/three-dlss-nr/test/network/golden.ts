// Golden hashes of the whole network on the synthetic model (seed 1) and `syntheticFeatures` (seed 1).
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Produced by the Chrome parity gate
// (test/browser/run-network-parity-chrome.mjs), which runs the reference WebGPU port and this port on one device in
// real Chrome and requires every tensor below to be byte-identical between them; it prints this table. Node tests
// (network.synthetic.gpu.test.ts) then compare our network alone against it, on any device: the TSL port never uses
// f16 hardware, so its bytes do not depend on the device.

/** What the parity checks read: the captured boundaries, the three post tensors, and the f32 head. */
export interface ParityReader {
  readonly boundaryNames: string[];
  readBoundary(name: string): Promise<Uint8Array>;
  readTensor(label: string): Promise<Uint8Array>;
  readHead(): Promise<Float32Array>;
}

/** The post-network tensors compared by label (unique labels in both graphs). */
export const POST_TENSORS = ['post merge', 'post merge raw', 'post block raw'] as const;

export interface NetworkDigest {
  /** Tensor name (`block-N`, ..., `post merge`, ..., `head`) -> SHA-256 hex prefix (16 digits). */
  tensors: Record<string, string>;
  /** SHA-256 prefix over all tensor hashes in order. */
  all: string;
}

const hex = (buffer: ArrayBuffer): string =>
  [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, '0')).join('');

export async function sha256Prefix(bytes: Uint8Array): Promise<string> {
  return hex(await crypto.subtle.digest('SHA-256', bytes as unknown as BufferSource)).slice(0, 16);
}

/** Every parity tensor of a network, in order: boundaries (capture order), post tensors, head. */
export async function readParityTensors(network: ParityReader): Promise<Map<string, Uint8Array>> {
  const tensors = new Map<string, Uint8Array>();
  for (const name of network.boundaryNames) tensors.set(name, await network.readBoundary(name));
  for (const label of POST_TENSORS) tensors.set(label, await network.readTensor(label));
  const head = await network.readHead();
  tensors.set('head', new Uint8Array(head.buffer, head.byteOffset, head.byteLength));
  return tensors;
}

export async function digestTensors(tensors: Map<string, Uint8Array>): Promise<NetworkDigest> {
  const digest: Record<string, string> = {};
  for (const [name, bytes] of tensors) digest[name] = await sha256Prefix(bytes);
  const all = await sha256Prefix(new TextEncoder().encode(Object.values(digest).join(',')));
  return { tensors: digest, all };
}

export const networkDigest = async (network: ParityReader): Promise<NetworkDigest> =>
  digestTensors(await readParityTensors(network));

/** Golden digests by valid size (`WxH`). Regenerate with the Chrome parity gate (`--golden`). */
export const NETWORK_GOLDEN: Record<string, NetworkDigest> = {};
