import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const root = fileURLToPath(new URL('.', import.meta.url));

/** The reference WebGPU port's ES modules (git submodule, read-only), imported by tests as `@ref/<file>.js`. */
const referenceSource = `${root}reference/OpenDLSS-NR/ports/browser-webgpu/src/`;

/**
 * Dawn options for the `gpu` project, from the environment:
 *   DLSS_NR_DAWN="backend=vulkan enable-dawn-features=allow_unsafe_apis"  (whitespace-separated), or
 *   DLSS_NR_DAWN_BACKEND=vulkan                                            (shorthand for backend=...).
 * Dawn picks the platform backend by default (D3D12 on Windows, Metal on macOS, Vulkan on Linux; lavapipe in CI).
 */
function dawnOptions(): string[] {
  const options = (process.env.DLSS_NR_DAWN ?? '').split(/\s+/).filter(Boolean);
  if (process.env.DLSS_NR_DAWN_BACKEND) options.push(`backend=${process.env.DLSS_NR_DAWN_BACKEND}`);
  return options;
}

// The reference implementation (git submodule) is never part of test discovery;
// tests import its modules explicitly.
const exclude = ['**/node_modules/**', '**/dist/**', 'reference/**'];

export default defineConfig({
  resolve: {
    // Tests (both projects) run against package sources, not dist builds.
    alias: [
      { find: 'three-dlss-nr', replacement: `${root}packages/three-dlss-nr/src/index.ts` },
      { find: /^@ref\//, replacement: referenceSource },
    ],
  },
  test: {
    coverage: {
      provider: 'v8',
      reportOnFailure: true,
      include: ['packages/three-dlss-nr/src/**/*.ts'],
      exclude: ['**/*.test.ts'],
      reporter: ['text', 'json-summary', 'lcov', 'html'],
    },
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          testTimeout: 30_000,
          // Generating the 141 MiB synthetic model (with SHA-256) in beforeAll takes
          // well over the 10 s default on CI runners under coverage instrumentation.
          hookTimeout: 120_000,
          include: ['packages/**/*.test.ts'],
          exclude: [...exclude, '**/*.gpu.test.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'gpu',
          // Headless WebGPU in Node via Google's Dawn (vitest-environment-webgpu-node):
          // navigator.gpu, GPU* globals and a headless canvas, no browser. See dawnOptions().
          environment: 'webgpu-node',
          environmentOptions: { webgpuNode: { dawnOptions: dawnOptions() } },
          // fetch() for file: and synthetic: URLs, so the reference port loads its WGSL and weights in Node.
          setupFiles: [`${root}packages/three-dlss-nr/test/setup/fetchShim.ts`],
          include: ['packages/**/*.gpu.test.ts'],
          exclude,
          testTimeout: 120_000,
          hookTimeout: 120_000,
        },
      },
    ],
  },
});
