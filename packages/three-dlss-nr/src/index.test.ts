import { describe, expect, it } from 'vitest';
import { UPSTREAM, VERSION } from './index.js';

describe('three-dlss-nr', () => {
  it('credits the upstream OpenDLSS-NR implementation at the pinned commit', () => {
    expect(UPSTREAM.url).toBe('https://github.com/maanHimself/OpenDLSS-NR');
    expect(UPSTREAM.commit).toBe('9d08f41');
    expect(UPSTREAM.license).toBe('MIT');
  });

  it('exports a semver version', () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
