import { defineConfig } from 'vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import viteReact from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';

export default defineConfig({
  resolve: { tsconfigPaths: true },
  plugins: [
    nitroV2Plugin({
      preset: 'node-server',
      compatibilityDate: '2025-11-07',
    }),
    tailwindcss(),
    tanstackStart(),
    viteReact(),
  ],
});
