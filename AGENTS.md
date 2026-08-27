# Agent Instructions
> **Workflow**: Follow [autonomous-coding-agents/AGENTS.md](./autonomous-coding-agents/AGENTS.md).
> [!CAUTION]
> **Submodule Immutability**: NEVER edit files in git submodules (100% READ-ONLY, zero mutations; store diffs in `patches/<submodule-name>/`, wrappers in `scripts/`).
> [!IMPORTANT]
> **Git Commit & Push Policy**: NEVER run `git commit` or `git push` automatically. Keep all changes uncommitted in the working tree. Commit and push ONLY when explicitly asked/instructed by the user.
> [!IMPORTANT]
> **Relative Paths in Documentation**: NEVER use full/absolute filesystem paths (e.g. `/home/aiserver/...` or `file:///...`) in markdown documentation, roadmaps, or plans. ALWAYS use standard relative paths (e.g. `./...`, `../...`).

## Milestone Directive (`m` / `d`)
* When user triggers **`m`** (Milestone):
  1. Extract all `[DONE]` tasks from [`plans/next-enhancements.md`](./plans/next-enhancements.md).
  2. Move and document them into `docs/features/` as domain feature docs (e.g. `docs/features/core/`, `docs/features/comparison/`, `docs/features/ingestion/`).
  3. Remove completed `[DONE]` items from [`plans/next-enhancements.md`](./plans/next-enhancements.md) keeping the active plan lean.
  4. Update milestone status in [`plans/roadmaps/`](./plans/roadmaps/).
