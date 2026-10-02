// WebGPU usage and map-mode flags as plain numbers (values fixed by the WebGPU specification).
//
// Part of three-dlss-nr, a port to three.js (TSL / WebGPU) of OpenDLSS-NR by maan (MIT,
// https://github.com/maanHimself/OpenDLSS-NR, pinned at 9d08f41). TypeScript's DOM library types `GPUDevice` and
// friends but does not declare the `GPUBufferUsage` / `GPUTextureUsage` / `GPUMapMode` globals, so the flags the
// backends use are spelled out here instead of relying on the globals at type-check time.

export const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  QUERY_RESOLVE: 0x0200,
} as const;

export const TEXTURE_USAGE = {
  TEXTURE_BINDING: 0x04,
} as const;

export const MAP_MODE_READ = 0x0001;
