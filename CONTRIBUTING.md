# Contributing to Paper Butler

Thanks for your interest in contributing! This document covers local setup, coding
conventions, and the pull request process. For the full house style (coding standards,
testing requirements, commit rules, ECC agent orchestration), see [AGENTS.md](AGENTS.md) —
this file is a shorter, contributor-facing entry point into it.

## Table of Contents

- [Local Development Setup](#local-development-setup)
- [Coding Conventions](#coding-conventions)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Filing Issues](#filing-issues)
- [Cutting a Release](#cutting-a-release)
- [Claude Code Skills](#claude-code-skills)
- [Recommended Claude Code Plugins](#recommended-claude-code-plugins)

## Local Development Setup

1. Install dependencies:
   ```bash
   uv sync                # installs dependencies, including dev tools
   # or: pip install -e .
   ```
2. Run the backend (serves PDFs to the UI):
   ```bash
   uv run poe fastapi
   ```
3. In a separate terminal, run the frontend:
   ```bash
   uv run poe streamlit
   ```
4. Open the Streamlit URL it prints (typically `http://localhost:8501`) and click
   "Login with Google" to connect your own Google Drive.

Optional: to enable AI metadata generation locally, set an `HF_TOKEN` — see the
[README](README.md#optional-ai-metadata-generation) for how to create one.

Running into a certificate error on a corporate network? See the
[troubleshooting note in the architecture doc](docs/ARCHITECTURE.md#troubleshooting-corporate-network--vpn-tls-interception).

### Changelog generation (git-cliff)

This repo uses [git-cliff](https://git-cliff.org) to generate `CHANGELOG.md` from
conventional commits (`cliff.toml` at the repo root defines the mapping; see
[AGENTS.md §8](AGENTS.md) for this repo's commit types). It's a standalone Rust binary,
not a Python package, so it isn't installed via `uv`:

```bash
brew install git-cliff       # macOS
# or: cargo install git-cliff
```

Generate the changelog locally with:

```bash
git-cliff --config cliff.toml --unreleased
```

Commit messages are validated against `cliff.toml`'s types (`feat`, `fix`, `refactor`,
`chore`, `docs`, `test`, `ci`, `perf`) by a local `commit-msg` hook and, on PRs, by CI —
see [Pre-commit hooks](#pre-commit-hooks) below to install it.

### Pre-commit hooks

```bash
uvx prek install
uvx prek run --all-files

uvx pre-commit install
uvx pre-commit install --hook-type commit-msg
```

## Coding Conventions

All coding style, file organization, type hint/docstring, and input validation rules live in
[AGENTS.md §3](AGENTS.md#3-coding-style) — read it before opening a PR. Highlights:

- Prefer immutable data (`frozen=True` Pydantic models/dataclasses where practical).
- Many small, focused files (~200–400 lines typical, 800 max) over few large ones.
- Full type hints on new/modified functions; Google-style docstrings.
- Validate all external input at API boundaries via Pydantic models.

## Testing Requirements

See [AGENTS.md §4](AGENTS.md#4-testing-requirements) for the full TDD workflow and pytest
conventions. In short:

- **Minimum 80% coverage** for new and existing backend functions.
- Write the test first (RED), implement to pass it (GREEN), then refactor.
- Verify before committing:
  ```bash
  uv run poe test        # full suite
  uv run poe check       # ruff format/lint, pyrefly, vulture, skylos
  ```

## Pull Request Process

1. **Branch naming**: `{type}/{issue-number}-{short-slug}`, e.g. `fix/73-oversized-upload` or
   `docs/137-contributing-guide`. Branch off an up-to-date `main`:
   ```bash
   git fetch origin main
   git checkout -b {type}/{n}-{slug} origin/main
   ```
2. **Commit messages**: `<type>: <short description>`, imperative mood, lowercase, under 50
   characters in the subject (the `commit-msg` hook and CI hard-fail above 75). Types:
   `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`, `perf`. See
   [AGENTS.md §8](AGENTS.md#8-commit-message-rules--examples) for the full rules and
   examples of good vs. bad messages.
3. **Quality gates**: run `uv run poe check` (ruff format/lint, pyrefly, vulture, skylos) and
   `uv run poe test` before opening a PR. Broken checks block review.
4. **Open the PR**: reference the issue with `Closes #{n}` in the PR body (not a plain `#{n}`
   mention) so the issue auto-closes on merge. Include a test plan listing the checks you ran.
5. **Review expectations**: PRs are reviewed for correctness, adherence to AGENTS.md
   conventions, and test coverage. Address CRITICAL/HIGH review feedback before merge;
   MEDIUM/LOW items are at the reviewer's discretion.

## Filing Issues

See [AGENTS.md §9](AGENTS.md#9-github-issue-format--labels) for this repo's issue body format
(`## Background` / `## Proposed fix` / `## Why this matters`) and label taxonomy
(`type:*`/`area:*`/`difficulty:*`).

## Cutting a Release

Releases are cut manually via the `Release` GitHub Actions workflow
(`.github/workflows/release.yml`), triggered by `workflow_dispatch` — nothing is
auto-released on merge. Given a `bump` level (`patch`/`minor`/`major`), it:

1. Bumps the `version` field in `pyproject.toml`.
2. Runs [git-cliff](https://git-cliff.org) (`cliff.toml`) to prepend a new section to
   `CHANGELOG.md` from conventional commits since the last tag, and separately renders
   just that section as GitHub release notes.
3. Unless `dry_run` is set, commits the version bump and changelog, tags the commit
   (`v{version}`), pushes, and publishes a GitHub Release with the generated notes.

With `dry_run: true`, steps 1–2 still run so you can inspect the output, but nothing is
committed, tagged, pushed, or published — the generated `CHANGELOG.md` is uploaded as a
build artifact instead (download it from the run's **Artifacts** section).

From the GitHub UI: **Actions → Release → Run workflow**, choose `bump` and whether to
`dry_run`. From the CLI (requires `gh`, authenticated):

```bash
# dry run — generates CHANGELOG.md as a build artifact only
gh workflow run release.yml -f bump=patch -f dry_run=true

# real release — commits, tags, pushes, and publishes a GitHub Release
gh workflow run release.yml -f bump=minor -f dry_run=false
```

`workflow_dispatch` workflows only become runnable once they exist on the default branch,
so this workflow can only be triggered after it's merged to `main`.

## Claude Code Skills

This repo ships two [Claude Code](https://claude.com/claude-code) skills under
`.claude/skills/` that automate parts of this contribution workflow:

- **`solve-issue`** — given an issue number or URL, fetches the issue, classifies its
  change shape from labels/body, implements it (via the matching `ecc:orch-*`
  orchestration skill for code changes, or directly for docs/trivial chores), opens a
  draft PR with `Closes #{n}`, and comments back on the issue linking to the PR.
- **`jules-feedback`** — triages Jules' (the automated PR-review bot) latest review
  comment on a PR into fixed code, technical pushback where Jules is wrong, and tracked
  GitHub issues for anything out of scope.

Both are optional tooling for contributors using Claude Code — they encode this file's
branch-naming, commit, and PR conventions so you don't have to apply them by hand.

## Recommended Claude Code Plugins

If you're contributing with [Claude Code](https://claude.com/claude-code), these two
plugin marketplaces cover most of the workflow described above and are what AGENTS.md's
agent/skill references (`ecc:*`, `superpowers:*`) resolve to:

- **[ECC](https://github.com/affaan-m/ECC)** — the agent/skill/rule set AGENTS.md is built
  on (planner, tdd-guide, python-reviewer, security-reviewer, and the `ecc:orch-*`
  orchestration skills used by `solve-issue`).
- **[Superpowers](https://github.com/obra/superpowers)** — general-purpose process skills
  (brainstorming, systematic debugging, TDD, writing plans, git worktrees) that ECC's
  `orch-*` pipelines and this repo's own skills lean on.

Install either via the Claude Code plugin marketplace UI, or see each repo's README for
manual setup.
