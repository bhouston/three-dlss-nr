// The vendored reference (vendor/opendlss-nr, built by scripts/bundle-reference.mjs) is the upstream code, byte for
// byte, at the pinned commit of OpenDLSS-NR by maan (MIT).

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { SHADERS, SOURCE } from '../../vendor/opendlss-nr/index.js';
import { UPSTREAM } from '../index.js';

const packageRoot = fileURLToPath(new URL('../..', import.meta.url));
const vendorRoot = join(packageRoot, 'vendor/opendlss-nr');
const submodule = join(packageRoot, '../../reference/OpenDLSS-NR');
// @ts-expect-error - plain .mjs build script without declarations
const bundler = await import('../../scripts/bundle-reference.mjs');

/** A file at the pinned commit (not the working tree), via git. */
const atPinnedCommit = (path: string): string =>
  execFileSync('git', ['-C', submodule, 'show', `${bundler.UPSTREAM_COMMIT}:${path}`], {
    encoding: 'utf8',
    env: bundler.gitEnv(),
    maxBuffer: 16 << 20,
  });

const hasSubmodule = existsSync(join(submodule, 'ports/browser-webgpu/src/network.js'));

describe('vendored OpenDLSS-NR reference', () => {
  it('pins the commit the package attributes', () => {
    expect(bundler.UPSTREAM_COMMIT.startsWith(UPSTREAM.commit)).toBe(true);
    expect(SOURCE.commit).toBe(bundler.UPSTREAM_COMMIT);
    expect(SOURCE.repository).toBe(UPSTREAM.url);
    expect(readFileSync(join(vendorRoot, 'LICENSE'), 'utf8')).toContain('Copyright (c) 2026 maan');
  });

  it.skipIf(!hasSubmodule)('inlines every WGSL file exactly as at the pinned commit', () => {
    expect(Object.keys(SHADERS).toSorted()).toEqual([...bundler.WGSL_FILES].toSorted());
    for (const name of bundler.WGSL_FILES) {
      expect(SHADERS[name], name).toBe(atPinnedCommit(`ports/browser-webgpu/shaders/${name}`));
    }
  });

  it.skipIf(!hasSubmodule)('copies every module reachable from network.js verbatim behind the header', () => {
    const modules: string[] = bundler.reachableModules(join(bundler.portRoot, 'src'));
    expect(modules).toEqual(expect.arrayContaining(['network.js', 'graph.js', 'matmul/index.js', 'window/index.js']));
    expect(modules).not.toContain('parity.js');
    expect(Object.keys(SOURCE.files)).toHaveLength(modules.length + bundler.WGSL_FILES.length);
    for (const path of modules) {
      const upstreamPath = `ports/browser-webgpu/src/${path}`;
      const vendored = readFileSync(join(vendorRoot, 'src', path), 'utf8');
      const header = bundler.vendorHeader(upstreamPath);
      expect(vendored.startsWith(header), path).toBe(true);
      expect(vendored.slice(header.length), path).toBe(atPinnedCommit(upstreamPath));
    }
  });

  it.skipIf(!hasSubmodule)('is up to date with the generator', () => {
    const generated: Map<string, string> = bundler.vendoredFiles();
    for (const [path, text] of generated) {
      expect(readFileSync(join(vendorRoot, path), 'utf8'), path).toBe(text);
    }
  });
});
