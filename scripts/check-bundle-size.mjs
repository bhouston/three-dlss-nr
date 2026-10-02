import { build } from 'esbuild';
import { gzipSync } from 'node:zlib';
import { appendFileSync } from 'node:fs';

// Measure complete package entry points with three.js supplied by the consumer.
//
// Budgets: the main entry is the whole network (451 dispatches from ~10 kernel factories written in TSL, the graph,
// the model loader, the frame kernels and the CPU-built lookup tables); 27.5 kB gzip before the attention kernels,
// which add roughly 6 kB. 40 kB leaves headroom for small fixes without hiding a regression such as a bundled copy of
// three or of the synthetic generator. `three-dlss-nr/synthetic` (weight and feature generator) stays separate.
const entries = [
  { name: 'three-dlss-nr', file: 'packages/three-dlss-nr/dist/index.js', limit: 40000 },
  { name: 'three-dlss-nr/synthetic', file: 'packages/three-dlss-nr/dist/synthetic/index.js', limit: 8000 },
];
const rows = ['| Entry | Minified gzip | Limit |', '| --- | ---: | ---: |'];
for (const { name, file, limit } of entries) {
  const result = await build({
    entryPoints: [file],
    bundle: true,
    minify: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    external: ['three', 'three/*'],
  });
  const bytes = gzipSync(result.outputFiles[0].contents).byteLength;
  rows.push(`| ${name} | ${bytes} B | ${limit} B |`);
  if (bytes > limit) process.exitCode = 1;
}
const summary = `${rows.join('\n')}\n`;
console.log(summary);
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
