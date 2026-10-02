#!/usr/bin/env node
// Rewrite the parity results' READMEs from the committed exactness.json and timing.json files (no GPU needed).
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT).

import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

import { suiteRoot, writeReadmes } from './suite.mjs';

const files = writeReadmes();
execFileSync(process.execPath, [join(suiteRoot, '../../node_modules/oxfmt/bin/oxfmt'), ...files], { stdio: 'inherit' });
