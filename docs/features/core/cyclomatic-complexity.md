# 🔍 Cyclomatic Complexity Skill & Automated Codebase Audit

> **Domain**: Code Quality, Maintainability & Refactoring  
> **Submodule**: [`cyclomatic-complexity-skill`](../../../cyclomatic-complexity-skill)  
> **Reference**: [saurabhkumar8112/cyclomatic-complexity-skill](https://github.com/saurabhkumar8112/cyclomatic-complexity-skill.git)  
> **Status**: `ACTIVE` ✅  

---

## 1. Overview

The **Cyclomatic Complexity Skill** enforces readable, clean, and human-maintainable code by measuring McCabe decision point density ($CC = \text{decisions} + 1$) across Python and polyglot source trees.

### Complexity Rating Scale:
* **Grade A (1–5)**: Clean, low complexity. Leave alone.
* **Grade B (6–10)**: Moderate complexity. Refactor if modifying nearby code.
* **Grade C (11–15)**: High complexity. Target for extraction into named units/predicates.
* **Grade D/F (> 15)**: Dangerous branching. Split into guard clauses and polymorphic handlers.

---

## 2. CLI Runner (`scripts/check_cyclomatic_complexity.py`)

A fast, zero-external-dependency AST analyzer and `radon` integration for automated CI/CD audits:

```bash
# Scan codebase with default threshold (max: 10, fail: 15)
uv run python scripts/check_cyclomatic_complexity.py

# Strict mode: fail exit code if any function exceeds threshold
uv run python scripts/check_cyclomatic_complexity.py --strict --fail-threshold 15

# Export JSON metrics
uv run python scripts/check_cyclomatic_complexity.py --json
```

---

## 3. Submodule Immutability

Per the repository's agent rules, `cyclomatic-complexity-skill/` is maintained as a **100% READ-ONLY** git submodule. All CLI wrappers, test suites, and configurations are hosted in:
* Tool CLI: [`scripts/check_cyclomatic_complexity.py`](../../../scripts/check_cyclomatic_complexity.py)
* Test Suite: [`tests/test_cyclomatic_complexity.py`](../../../tests/test_cyclomatic_complexity.py)
* Global Agent Skill: `~/.gemini/config/skills/cyclomatic-complexity/SKILL.md`
