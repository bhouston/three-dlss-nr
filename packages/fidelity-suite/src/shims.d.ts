// three ships no TypeScript declarations (same shorthand modules as the library), and the reference port's modules
// (`@ref/<file>.js`, OpenDLSS-NR's WebGPU port by maan, MIT, bundled through an esbuild alias) are plain JavaScript.
declare module 'three/webgpu';
declare module 'three/tsl';
declare module 'three/addons/*';
declare module '@ref/*';
