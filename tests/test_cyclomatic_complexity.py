"""Tests for cyclomatic complexity analyzer and audit tool."""

import ast
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_cyclomatic_complexity import ComplexityVisitor, analyze_file, get_grade, scan_directory


def test_complexity_visitor_simple_function():
    """Verify complexity of a simple linear function is 1."""
    code = """
def add(a, b):
    return a + b
"""
    tree = ast.parse(code)
    visitor = ComplexityVisitor("test.py")
    visitor.visit(tree)
    assert len(visitor.results) == 1
    assert visitor.results[0]["name"] == "add"
    assert visitor.results[0]["complexity"] == 1


def test_complexity_visitor_branching_function():
    """Verify decision point calculation with branches and boolean ops."""
    code = """
def evaluate(x, y, z):
    if x > 0 and y > 0:
        return x + y
    elif x < 0 or z < 0:
        return x - y
    for i in range(5):
        if i == z:
            break
    return 0
"""
    tree = ast.parse(code)
    visitor = ComplexityVisitor("test.py")
    visitor.visit(tree)
    assert len(visitor.results) == 1
    assert visitor.results[0]["name"] == "evaluate"
    # 1 (base) + 1 (if) + 1 (and) + 1 (elif) + 1 (or) + 1 (for) + 1 (if) = 7
    assert visitor.results[0]["complexity"] == 7


def test_get_grade():
    """Verify letter grade classification."""
    assert get_grade(3) == "A (Low)"
    assert get_grade(8) == "B (Moderate)"
    assert get_grade(15) == "C (High)"
    assert get_grade(25) == "D (Very High)"
    assert get_grade(35) == "F (Dangerous)"


def test_scan_directory_src():
    """Test scanning src/transcribe directory."""
    blocks, warns, fails = scan_directory(["src/transcribe"], max_complexity=10, fail_threshold=15)
    assert len(blocks) > 50
    assert isinstance(warns, int)
    assert isinstance(fails, int)
