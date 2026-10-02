// `three` ships no TypeScript declarations for its webgpu/tsl subpaths in this
// repo (see packages/three-dlss-nr/src/three-shims.d.ts) - every import from
// these resolves to `any`.
declare module 'three';
declare module 'three/webgpu';
declare module 'three/tsl';
declare module 'three/addons/*';

declare module '*.css?url' {
  const url: string;
  export default url;
}
