# Contributing

These rules apply to every contributor, human or AI agent. This is the source of truth for the contribution workflow. `AGENTS.md` and `CLAUDE.md` point here.

## Issue → branch → pull request

1. **Start with an issue.** Describe the problem, motivation, constraints, and testable acceptance criteria. Reuse an existing issue when it covers the work.
2. **Branch from `main`.** Fetch current `origin/main` and name the branch `<type>/<issue>-<short-description>`, such as `feat/42-feature-block-port`. Use a separate worktree for unrelated local changes.
3. **Commit with Conventional Commits.** Reference the issue in the body where useful.
4. **Run the local checks** below before opening a PR.
5. **Open a PR against `main`.** Use a Conventional Commit title, include `Closes #<issue>` in the body, explain the resulting behavior, and report validation.
6. **Merge on green CI.** Use a merge commit; do not squash or rebase merge. Maintainers choose when to merge.

`main` is the only integration branch. The initial repository bootstrap can land directly on `main`; subsequent tracked changes use PRs.

## Commit format

Use `type(optional-scope): description` in the imperative mood. Allowed types: `feat`, `fix`, `perf`, `docs`, `chore`, `refactor`, `test`, `style`, `build`, `ci`, and `revert`.

- `feat:` produces a minor release.
- `fix:` and `perf:` produce a patch release.
- `!` after the type/scope or a `BREAKING CHANGE:` footer produces a major release.
- Other types do not trigger a release on their own.

After `pnpm install`, Husky runs commitlint on commits. CI checks the PR title and commits; Git-generated merge commits are exempt.

## Local checks

Use the Node version in `.nvmrc` and the pnpm version in `package.json`. The reference implementation is a git submodule, so clone with `--recurse-submodules` or run `git submodule update --init` first.

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm tsc
pnpm lint
pnpm format:check
pnpm test
pnpm test:gpu
pnpm release:check
pnpm size
```

`pnpm test:gpu` runs the `*.gpu.test.ts` files on headless WebGPU in Node (Google's Dawn, via `vitest-environment-webgpu-node`), using the machine's GPU. Linux without a GPU needs Mesa's software Vulkan driver: `sudo apt-get install -y libegl1 libgles2 libgl1-mesa-dri mesa-vulkan-drivers` and `LIBGL_ALWAYS_SOFTWARE=1`. The pre-commit hook formats and lints staged files and type-checks the workspace. CI runs the checks on Linux, with the GPU tests on lavapipe. Keep the network port in `packages/three-dlss-nr` and the demo in `packages/website`. Treat `reference/OpenDLSS-NR` as read-only: it is pinned upstream code used for parity tests. Never download, extract, or commit NVIDIA model weights; tests use deterministic synthetic weights. Ported files keep a header crediting OpenDLSS-NR by maan.

## Releases

Merging to `main` never publishes. A maintainer dispatches the release workflow when accumulated changes should ship:

```sh
gh workflow run release.yml --ref main
gh workflow run release.yml --ref main -f dry_run=true
```

The workflow requires `main`, reruns CI on the selected commit, and uses semantic-release to select the version from Conventional Commits. It publishes `three-dlss-nr` through npm trusted publishing (GitHub environment `npm`). The website is private and deploys to Cloud Run from `main`.

## Security

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/bhouston/three-dlss-nr/security/advisories/new), as described in [SECURITY.md](SECURITY.md). Do not disclose exploit details in a public issue.
