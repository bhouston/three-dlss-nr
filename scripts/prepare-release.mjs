import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

export const packages = ['three-dlss-nr'];

// Only the CI checkout is modified. Git tags are the version source of truth.
//
// This module runs as two separate "prepare" plugin entries in
// release.config.js, wrapped around @anolilab/semantic-release-pnpm's own
// `prepare` entry:
//
//   1. files (default): stage LICENSE/NOTICE/CHANGELOG.md into the package dir
//      and register CHANGELOG.md in `files`, before pnpm bumps the version and
//      packs/publishes.
//   2. artifacts (`{ artifacts: true }`, run last): once the package has been
//      version-bumped, pack it into release-artifacts/ for the GitHub release.
//      `pnpm pack <dir>` ignores a directory argument and always packs the
//      current working directory's package, so `pnpm --dir <pkgRoot> pack` is
//      used instead of the plugin's `tarballDir` option.
export function prepare({ artifacts } = {}, { cwd, env }) {
  if (artifacts) {
    const packDestination = join(cwd, 'release-artifacts');
    for (const name of packages) {
      execFileSync('pnpm', ['--dir', join(cwd, 'packages', name), 'pack', '--pack-destination', packDestination], {
        cwd,
        env,
        stdio: 'pipe',
      });
    }
    return;
  }
  for (const name of packages) {
    const dir = join(cwd, 'packages', name);
    const path = join(dir, 'package.json');
    const pkg = JSON.parse(readFileSync(path, 'utf8'));
    pkg.files = [...new Set([...pkg.files, 'CHANGELOG.md'])];
    writeFileSync(path, `${JSON.stringify(pkg, null, 2)}\n`);
    copyFileSync(join(cwd, 'LICENSE'), join(dir, 'LICENSE'));
    copyFileSync(join(cwd, 'NOTICE'), join(dir, 'NOTICE'));
    copyFileSync(join(cwd, 'CHANGELOG.md'), join(dir, 'CHANGELOG.md'));
  }
}
