// `three` ships no TypeScript declarations of its own, and the closest
// published `@types/three` release lags the Node-material / TSL API this
// package uses. Shorthand ambient modules: every import from 'three' /
// 'three/tsl' / 'three/webgpu' resolves to `any` (same approach as three-ntc).
declare module 'three';
declare module 'three/tsl';
declare module 'three/webgpu';
