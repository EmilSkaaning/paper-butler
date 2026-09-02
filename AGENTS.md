# Agent Developer Guidelines

This document governs how work is done in the `paper-butler` repository — coding style,
testing, and how function/class documentation and Git commits are carried out. Sections 1–4 and
7–8 are adapted from [ECC](https://github.com/affaan-m/ECC)'s `AGENTS.md`/`CLAUDE.md`, pruned to
this repo's actual stack (a single Python project: FastAPI backend, Streamlit frontend,
pytest/ruff/pyrefly via `uv`).

## 0. Branding

UI colors, fonts, and iconography follow [docs/BRANDING.md](docs/BRANDING.md).
Consult it before touching `.streamlit/config.toml`, `frontend/branding.py`,
or adding new colors/fonts/icons to the Streamlit UI.

## 1. Core Principles

1. **Agent-First** — delegate to ECC's specialized agents/skills for domain tasks (see Section 5).
2. **Test-Driven** — write tests before implementation; 80%+ coverage required (see Section 4).
3. **Security-First** — never compromise on security; validate all inputs (see Section 3 and Section 6).
4. **Immutability** — prefer creating new objects over mutating existing ones (see Section 3).
5. **Plan Before Execute** — plan complex features or refactors before writing code (`ecc:plan`).

## 2. Environment

### Python Virtual Environment
* Always execute python commands and tools using `uv` to ensure proper environment isolation.
* Example: `uv run pytest` or `uv run ruff format .`

### Git Worktrees
* All git worktrees live under `.worktrees/` at the repo root (already gitignored) — never
  create one as a sibling of the repo or elsewhere on disk.
* Name each worktree directory after its branch, dropping the `type/` prefix, e.g. branch
  `fix/73-oversized-upload` → `.worktrees/73-oversized-upload`.
* Create with: `git worktree add .worktrees/{name} -b {type}/{name} origin/main`.

### Plan Files
* Longer implementation plans or future-scoping/roadmap plans (multi-step features,
  epics, release evaluations, architecture proposals) should be written to `plans/` at
  the repo root — it is gitignored, so files there stay local and don't need to be
  committed, but are easy to find again in the working directory.

## 3. Coding Style

**Immutability:** Prefer returning new objects/copies over mutating in place — e.g. Pydantic
models and dataclasses should default to immutable (`frozen=True`) where practical, and functions
should return new values rather than mutate their arguments.

**File organization:** Many small, focused files over few large ones — ~200–400 lines typical,
800 max. Organize by feature/domain (`backend/`, `frontend/`), not by type.

**Error handling:** Handle errors at every level. Provide user-friendly messages in the Streamlit
UI; log detailed context server-side in the FastAPI backend. Never silently swallow errors.

**Code quality checklist:** Functions small (<50 lines); no nesting deeper than 4 levels; no
hardcoded values; readable, well-named identifiers.

### Type Hints & Docstrings (`func-documentation` skill)
* **Type Hints**: Add or update type hints for any new or modified functions/classes. All variables and functions must be fully type-hinted. Avoid `typing.Any` — use it only when no more specific type is possible (e.g. truly dynamic third-party data), and justify its use with an inline comment.
* **Docstrings**: Provide clean docstrings describing parameters, return values, and exceptions for any new or modified functions/classes. **Use the Google docstring format strictly.**
* **Input Verification**: Use Pydantic models for incoming data structures and API endpoint validation.
* **Exclusions**: Skip files where the changes are purely deletions or trivial (e.g. config, constants, `__init__.py`).

## 4. Testing Requirements

**Minimum coverage:** 80% for all new and existing backend functions.

**Required test types:**
1. **Unit tests** — individual functions, utilities, components.
2. **Integration tests** — FastAPI endpoints, Google Drive / HuggingFace client boundaries.

E2E/browser tests are not currently part of this repo's toolchain (no Playwright/Selenium setup)
— add them only if that infrastructure is introduced later.

**TDD workflow (mandatory):**
1. Write the test first (RED) — it should FAIL.
2. Write the minimal implementation (GREEN) — it should PASS.
3. Refactor (IMPROVE) — verify coverage is still 80%+.

**Troubleshooting failures:** check test isolation → verify mocks → fix the implementation (not
the test, unless the test itself is wrong).

**Pytest conventions:**
* **Structured Tests**: Group similar tests under test classes.
* **Parametrization**: Use `@pytest.mark.parametrize` for screening over large settings to keep test code clean.
* **Fixtures**: Information used multiple times should be extracted into `conftest.py` files as pytest fixtures.
* **Mocking**: Use `pytest-mock` for mocking external dependencies (e.g., file system, APIs).
* **Execution**: Verify your changes by running unit tests before making any commit:
  ```bash
  uv run poe test
  # or
  uv run pytest
  ```

---

## 5. ECC Agent Orchestration

The [ECC](https://github.com/affaan-m/ECC) plugin marketplace is enabled for this repository and
provides specialized agents/skills. Use them proactively for the domain tasks below — they
supplement the `commit-code` workflow in Section 7 (quality gates, staging discipline, and
commit-message rules in Section 8 still govern the actual commit; ECC does not replace any of
that).

| Task | ECC skill / agent |
|------|--------------------|
| Plan a feature or refactor | `ecc:plan` skill / `ecc:planner` agent |
| Architecture or scalability decisions | `ecc:architect` agent |
| TDD (red-green-refactor) | `ecc:tdd-guide` agent, `ecc:python-testing` / `ecc:tdd-workflow` skills |
| Python code review | `ecc:python-review` skill / `ecc:python-reviewer` agent |
| FastAPI backend review | `ecc:fastapi-review` skill / `ecc:fastapi-reviewer` agent |
| Security review | `ecc:security-review` skill / `ecc:security-reviewer` agent |
| Build or type errors | `ecc:build-fix` skill / `ecc:build-error-resolver` agent |
| Dead code cleanup | `ecc:refactor-clean` skill / `ecc:refactor-cleaner` agent |
| Docs or codemaps | `ecc:update-docs` / `ecc:update-codemaps` skills / `ecc:doc-updater` agent |

Before Step 3 (Quality checks) of `commit-code`, run the reviewer skill(s) relevant to the files
touched (`ecc:python-review`, `ecc:fastapi-review`, `ecc:security-review`) as an additional pass —
this is on top of the mandatory `ruff`/`pyrefly` gates, not a substitute for them.

## 6. Security Guidelines

* **No hardcoded secrets**: Google OAuth client secrets, API keys, and tokens must never be
  committed. `token.json` and `credentials.json` stay gitignored.
* **Boundary validation**: All external input is validated via Pydantic models at API boundaries
  (see Section 3's "Input Verification").
* **No sensitive data in errors**: Error messages returned to the FastAPI or Streamlit UI must not
  leak file paths, credentials, or stack traces.
* **On a suspected leak or vulnerability**: stop, run `ecc:security-review`, fix CRITICAL/HIGH
  findings, and rotate any exposed secrets before continuing other work.

## 7. Core Workflow: commit-code

Use this workflow whenever committing changes or saving work.

### Step 1 — Understand the changes
The agent must run `git diff` and `git status` in parallel to understand what changed and why. The agent should read any relevant modified files if the diff alone is not enough to write a precise commit message.

### Step 1b — Decide whether to split into multiple commits
The agent must inspect the full diff across all changed files (not just one file at a time) and group changes by logical concern.
* The agent should split changes into multiple commits when the diff contains *two or more genuinely unrelated logical changes* (e.g. a bug fix in one module plus a new feature in another).
* The agent must *not* split:
  * A single feature/fix that happens to touch several files
  * Small incidental changes tightly coupled to the main change (e.g. a helper used by the new code)
If a split makes sense, the agent must inform the user of the proposed grouping and commit order before proceeding, and repeat Steps 2–6 (stage, message, commit) once per group. If everything belongs to one logical change, the agent should proceed as a single commit.

### Step 2 — Update documentation
For each changed Python file, the agent must apply the `func-documentation` standards described in Section 3. The agent must re-run `git diff` after this step to ensure documentation changes are staged together with the code changes in Step 4.

### Step 3 — Quality checks
The agent must run all checks via the aggregate `poe` task:
```bash
uv run poe check
```
This runs, in order: `ruff-format-check`, `ruff-check`, `pyrefly-check`, `vulture-check`, and
`skylos-check`. If Ruff, Pyrefly, or Vulture fail, the agent must report the errors to the user and
stop — the agent must not commit broken code. Skylos (`skylos . -a --confidence 80`) runs in
report-only mode and is not a merge gate — review anything it surfaces in files you touched and
route it to the tracked Skylos backlog issue rather than blocking the commit on it.

If checks fail, the agent should offer to auto-fix what can be fixed automatically:
* `uv run poe ruff-format` — fixes formatting
* `uv run poe ruff-check-fix` — fixes auto-fixable lint issues
* Pyrefly and Vulture errors must be fixed manually.

The agent must re-run `uv run poe check` after any auto-fix before proceeding.

### Step 4 — Stage files
The agent must stage only the files relevant to the logical change, including any documentation files updated in Step 2. The agent should prefer explicit file paths over `git add .` to avoid accidentally including unrelated or sensitive files.
If untracked files exist that are unrelated to the change, the agent must leave them unstaged.

### Step 5 — Write the commit message
The agent must use ECC's commit format:
```
<type>: <short description>

[optional body]
```
* **Types**: feat, fix, refactor, chore, docs, test, ci, perf
* **Short description**: imperative mood, lowercase, no trailing period

The agent must construct the message, then show it to the user for confirmation before committing.

### Step 6 — Commit
The agent must execute the commit using the agreed message:
```bash
git commit -m "$(cat <<'EOF'
type: short description

Optional body here.
EOF
)"
```
The agent must not push unless the user explicitly asks. If splitting into multiple commits (Step 1b), the agent must repeat Steps 4–6 for each remaining group before finishing.

**PR workflow (when a PR is requested):** analyze the full commit history for the branch → draft a
comprehensive summary → include a test plan → push with the `-u` flag.

## 8. Commit Message Rules & Examples

### Hard Rules — Never Break These:

| Rule | Reason |
|------|--------|
| Keep the subject under 50 chars | Git truncates it in logs and UIs (CI hard-fails above 75) |
| Never use generic messages | "update code", "fix bug", "changes", "wip", "misc" tell reviewers nothing |
| Never commit unrelated files together | One logical change per commit keeps history bisectable |
| Never skip quality checks | Broken commits block the team and CI |
| Never commit secrets or credentials | Check .env, config files before staging |

### Examples

* **Good**:
  * `feat: enumerate all 15 alphafold model weights`
  * `fix: correct tissue mask path for wt samples`
  * `refactor: replace session tokens with short-lived JWTs`
  * `chore: bump modal to 0.67.0`

* **Bad**:
  * `update code` (generic, meaningless)
  * `fix bug` (which bug? in what?)
  * `WIP` (not a commit)
  * `changed some stuff` (tells reviewers nothing)
  * `feat: implemented the new de novo alphafold model weight enumeration system` (way over 50 chars)

## 9. GitHub Issue Format & Labels

When filing a new issue in this repo (by hand or via an agent), follow the structure and
labeling scheme already used by existing issues.

### Standard issue body format

Every regular (non-epic) issue body must use exactly these three headers, in this order:

* `## Background` — what's wrong, missing, or motivating the issue.
* `## Proposed fix` — the concrete change being requested.
* `## Why this matters` — the impact of leaving it undone.

### Epic issue body format

For work too large for one issue (e.g. epics #41, #42, #43), file a single epic issue plus one
regular issue per sub-task, and link them together:

* Epic issue body headers, in this order:
  * `## Problem` — what's wrong or missing at the epic level.
  * `## Goal` — the end state once the whole epic ships.
  * `## Design` (optional) — cross-cutting architecture/decisions that apply to multiple
    sub-tasks, so each sub-task issue doesn't have to repeat them.
  * `## Tasks` — a GitHub checklist referencing each sub-task issue, e.g. `- [ ] #57 <short
    description>`. Check items off as sub-task issues close.
  * `## Cross-epic dependency` (if applicable) — call out hard dependencies on other epics or
    specific sub-task issues, and which remaining tasks are unblocked vs. blocked.
* Each sub-task issue is a normal `## Background` / `## Proposed fix` / `## Why this matters`
  issue as above, and should reference its parent epic inline (e.g. "see epic #42") the first
  time epic-level context is needed.
* There is no dedicated `epic:*` label — an epic is just a `type:feature` (or `type:chore`)
  issue identified by having a `## Tasks` checklist of sub-issues; label it and its sub-tasks
  normally per the taxonomy below.

### Label taxonomy

Apply **one `type:*`**, **one or more `area:*`**, and **one `difficulty:*`** label to every
new issue:

| Label | Meaning |
|-------|---------|
| `type:feature` | A concrete new capability to build |
| `type:chore` | Maintenance/tooling/asset work |
| `type:docs` | Documentation-only changes (README, guides, architecture docs) |
| `type:investigation` | Open-ended research/spike, not yet a committed feature |
| `area:backend` | FastAPI/data/model layer (`src/backend`) |
| `area:frontend` | Streamlit UI (`src/frontend`) |
| `area:ci-cd` | GitHub Actions, pre-commit, deployment |
| `area:design` | Visual/branding assets |
| `difficulty:trivial` | Minutes of work, no design decisions |
| `difficulty:easy` | Small, well-scoped change with a clear approach |
| `difficulty:medium` | Moderate scope, touches a few files or needs some design choices |
| `difficulty:hard` | Large or ambiguous scope, spans multiple layers/files |
| `difficulty:expert` | Open-ended architecture/strategy work with significant unknowns |

An issue missing any of the three label categories should be edited to add the missing
label(s) before work starts on it.
