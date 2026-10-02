import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const root = fileURLToPath(new URL('.', import.meta.url));

// The reference implementation (git submodule) is never part of test discovery;
// tests import its modules explicitly.
const exclude = ['**/node_modules/**', '**/dist/**', 'reference/**'];

export default defineConfig({
  resolve: {
    // Tests (both projects) run against package sources, not dist builds.
    alias: {
      'three-dlss-nr': `${root}packages/three-dlss-nr/src/index.ts`,
    },
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
          include: ['packages/**/*.test.ts'],
          exclude: [...exclude, '**/*.gpu.test.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'gpu',
          // Headless WebGPU in Node via Google's Dawn (vitest-environment-webgpu-node):
          // navigator.gpu, GPU* globals and a headless canvas, no browser. Dawn picks
          // the platform backend (D3D12 on Windows, Metal on macOS, Vulkan on Linux;
          // lavapipe in CI). Override with DLSS_NR_DAWN_BACKEND, e.g. `vulkan`.
          environment: 'webgpu-node',
          environmentOptions: {
            webgpuNode: {
              dawnOptions: process.env.DLSS_NR_DAWN_BACKEND ? [`backend=${process.env.DLSS_NR_DAWN_BACKEND}`] : [],
            },
          },
          include: ['packages/**/*.gpu.test.ts'],
          exclude,
          testTimeout: 120_000,
          hookTimeout: 120_000,
        },
      },
    ],
  },
});
