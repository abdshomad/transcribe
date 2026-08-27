#!/usr/bin/env python3
"""CLI utility to measure and audit Cyclomatic Complexity across codebase files."""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to compute McCabe Cyclomatic Complexity per function/method/class."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.results: List[Dict[str, Any]] = []

    def _calculate_node_complexity(self, node: ast.AST) -> int:
        """Count decision points + 1."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each 'and' / 'or' adds a branch decision point
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.IfExp, ast.Assert)):
                complexity += 1
        return complexity

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        cc = self._calculate_node_complexity(node)
        self.results.append({
            "type": "function",
            "name": node.name,
            "lineno": node.lineno,
            "complexity": cc,
            "filepath": self.filepath,
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        cc = self._calculate_node_complexity(node)
        self.results.append({
            "type": "async_function",
            "name": node.name,
            "lineno": node.lineno,
            "complexity": cc,
            "filepath": self.filepath,
        })
        self.generic_visit(node)


def analyze_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parse and calculate cyclomatic complexity for a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        visitor = ComplexityVisitor(str(filepath))
        visitor.visit(tree)
        return visitor.results
    except Exception as e:
        return [{
            "type": "error",
            "name": f"PARSE_ERROR: {e}",
            "lineno": 1,
            "complexity": 0,
            "filepath": str(filepath),
        }]


def get_grade(cc: int) -> str:
    """Return McCabe letter grade for given complexity score."""
    if cc <= 5:
        return "A (Low)"
    if cc <= 10:
        return "B (Moderate)"
    if cc <= 20:
        return "C (High)"
    if cc <= 30:
        return "D (Very High)"
    return "F (Dangerous)"


def scan_directory(
    target_dirs: List[str],
    max_complexity: int = 10,
    fail_threshold: int = 15,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Scan directories and collect complexity statistics."""
    all_blocks: List[Dict[str, Any]] = []

    for dir_path in target_dirs:
        p = Path(dir_path)
        if not p.exists():
            continue
        if p.is_file() and p.suffix == ".py":
            all_blocks.extend(analyze_file(p))
        elif p.is_dir():
            for py_file in p.glob("**/*.py"):
                if ".venv" in py_file.parts or "build" in py_file.parts or ".git" in py_file.parts:
                    continue
                all_blocks.extend(analyze_file(py_file))

    # Sort descending by complexity
    all_blocks.sort(key=lambda x: x["complexity"], reverse=True)

    warn_count = sum(1 for b in all_blocks if b["complexity"] > max_complexity)
    fail_count = sum(1 for b in all_blocks if b["complexity"] > fail_threshold)

    return all_blocks, warn_count, fail_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyclomatic Complexity Checker for Python codebase.")
    parser.add_argument("paths", nargs="*", default=["src/transcribe", "scripts"], help="Paths to analyze.")
    parser.add_argument("--max-complexity", "-m", type=int, default=10, help="Warning complexity threshold (default: 10).")
    parser.add_argument("--fail-threshold", "-f", type=int, default=15, help="Failure complexity threshold (default: 15).")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit with error if any function exceeds fail-threshold.")

    args = parser.parse_args()

    blocks, warns, fails = scan_directory(
        target_dirs=args.paths,
        max_complexity=args.max_complexity,
        fail_threshold=args.fail_threshold,
    )

    if args.json:
        print(json.dumps({
            "total_blocks": len(blocks),
            "warnings_count": warns,
            "failures_count": fails,
            "max_complexity_threshold": args.max_complexity,
            "fail_threshold": args.fail_threshold,
            "blocks": blocks,
        }, indent=2))
        return 1 if (args.strict and fails > 0) else 0

    print(f"\n📊 === Cyclomatic Complexity Audit ({len(blocks)} blocks analyzed) ===\n")
    print(f"{'Path':<40} {'Function/Method':<35} {'CC':<5} {'Grade'}")
    print("-" * 95)

    for b in blocks[:30]:  # Show top 30 hotspots
        rel_path = Path(b["filepath"]).as_posix()
        if len(rel_path) > 38:
            rel_path = "..." + rel_path[-35:]
        func_name = b["name"][:33]
        cc = b["complexity"]
        grade = get_grade(cc)
        color_flag = "🚨" if cc > args.fail_threshold else ("⚠️" if cc > args.max_complexity else "✅")
        print(f"{rel_path:<40} {func_name:<35} {color_flag} {cc:<3} {grade}")

    avg_cc = sum(b["complexity"] for b in blocks) / len(blocks) if blocks else 0
    print("-" * 95)
    print(f"Summary: Average CC: {avg_cc:.2f} | Warnings (> {args.max_complexity}): {warns} | Hotspots (> {args.fail_threshold}): {fails}\n")

    if args.strict and fails > 0:
        print(f"❌ Strict audit failed: {fails} block(s) exceed maximum threshold {args.fail_threshold}.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
