#!/usr/bin/env node
// Export the parity results as a static site into the website's public/parity, served at /parity/.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT). Runs `fidelity-kit build`, then pins the
// viewer's base URL: its index.html loads assets and data relative to the page, and the website's server also answers
// `/parity` (no trailing slash) with that page, where relative URLs would resolve against `/`. A `<base>` element
// makes both URLs work. Usage: node scripts/build.mjs [--out <dir>] [--base /parity/]

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { resultsDir, suiteRoot } from './suite.mjs';

const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const out = resolve(option('--out', join(suiteRoot, '../website/public/parity')));
const base = option('--base', '/parity/');

// The package's CLI (its exports do not expose the bin), linked into this package's node_modules by pnpm.
const cli = join(suiteRoot, 'node_modules/fidelity-kit/dist/bin.js');
execFileSync(process.execPath, [cli, 'build', resultsDir, '--out', out], { stdio: 'inherit' });

const index = join(out, 'index.html');
const html = readFileSync(index, 'utf8').replace(/<base [^>]*>\s*/, '');
writeFileSync(index, html.replace('<meta charset="utf-8" />', `<meta charset="utf-8" />\n    <base href="${base}" />`));
console.log(`parity site: ${out} (base ${base})`);
