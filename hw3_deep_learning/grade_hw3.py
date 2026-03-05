#!/usr/bin/env python3
"""
Grading script for HW3: Deep Learning.

Usage:
    python grade_hw3.py path/to/student_hw3_deep_learning.ipynb

The script:
1. Copies the student notebook into a temp directory alongside our helpers,
   tests, and test_data.
2. Executes the notebook (all cells run top-to-bottom).
3. After execution, imports the student's classes from the notebook namespace
   and runs our canonical test suite against them.
4. Counts passed/total tests and prints a percentage score.

Anti-cheat: the expected test list comes from TEST_REGISTRY in
dl_tests.py. Any test that doesn't run (e.g. because the student
deleted a test cell) counts as 0/1.
"""

import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path

# Import the canonical registry (this script lives next to dl_tests.py)
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from dl_tests import TEST_REGISTRY, TOTAL_EXPECTED_TESTS


def run_notebook_and_grade(notebook_path: Path, work_dir: Path) -> dict | None:
    """
    Execute the student notebook and run our test suite via an injected cell.

    Returns a dict mapping class_name -> list of (test_name, passed_bool),
    or None if extraction failed.
    """
    import nbformat
    from nbclient import NotebookClient

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Inject a grading cell that uses TEST_REGISTRY from our dl_tests
    grading_cell_source = '''
import json as _json
import matplotlib as _matplotlib
_matplotlib.use("Agg")

from dl_tests import TEST_REGISTRY

_grade_results = {}

for _name, _entry in TEST_REGISTRY.items():
    _cls = globals().get(_name)
    if _cls is None:
        _grade_results[_name] = []
        continue
    try:
        _results = _entry["check_fn"](_cls)
        _grade_results[_name] = [(tn, bool(tp)) for tn, tp, _ in _results]
    except Exception as _e:
        _grade_results[_name] = []

print("__GRADE_RESULTS_START__")
print(_json.dumps(_grade_results))
print("__GRADE_RESULTS_END__")
'''

    nb.cells.append(nbformat.v4.new_code_cell(source=grading_cell_source))

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    client.execute()

    # Extract results from the grading cell output
    for output in nb.cells[-1].get("outputs", []):
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if "__GRADE_RESULTS_START__" in text:
            lines = text.split("\n")
            capture = False
            for line in lines:
                if line.strip() == "__GRADE_RESULTS_START__":
                    capture = True
                    continue
                if line.strip() == "__GRADE_RESULTS_END__":
                    break
                if capture:
                    return json.loads(line.strip())

    print("ERROR: Could not extract grading results from notebook execution.")
    print("The notebook may have crashed before reaching the grading cell.")
    return None


def grade(results: dict) -> tuple[int, int, float]:
    """
    Grade based on canonical TEST_REGISTRY expectations.

    Returns (passed, total, percentage).
    """
    passed = 0
    total = TOTAL_EXPECTED_TESTS

    for class_name, entry in TEST_REGISTRY.items():
        actual_lookup = dict(results.get(class_name, []))
        for test_name in entry["tests"]:
            if actual_lookup.get(test_name, False):
                passed += 1

    percentage = (passed / total * 100) if total > 0 else 0.0
    return passed, total, percentage


def main():
    parser = argparse.ArgumentParser(
        description="Grade a student's HW3 deep learning notebook."
    )
    parser.add_argument(
        "notebook",
        type=str,
        help="Path to the student's hw3_deep_learning.ipynb",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-test results",
    )
    args = parser.parse_args()

    student_notebook = Path(args.notebook).resolve()
    if not student_notebook.exists():
        print(f"ERROR: File not found: {student_notebook}")
        sys.exit(1)

    # Our grading infrastructure lives alongside this script
    hw3_dir = Path(__file__).parent.resolve()

    # Create a temporary working directory with our infrastructure + student notebook
    with tempfile.TemporaryDirectory(prefix="hw3_grade_") as tmpdir:
        work_dir = Path(tmpdir)

        # Copy our infrastructure
        shutil.copy2(hw3_dir / "dl_helpers.py", work_dir / "dl_helpers.py")
        shutil.copy2(hw3_dir / "dl_tests.py", work_dir / "dl_tests.py")
        shutil.copytree(hw3_dir / "test_data", work_dir / "test_data")

        # Copy student notebook
        target_notebook = work_dir / "hw3_deep_learning.ipynb"
        shutil.copy2(student_notebook, target_notebook)

        print(f"Grading: {student_notebook.name}")
        print(f"Working directory: {work_dir}")
        print()

        # Execute and grade
        try:
            results = run_notebook_and_grade(target_notebook, work_dir)
        except Exception as e:
            print(f"ERROR: Notebook execution failed: {e}")
            print(f"\nScore: 0 / {TOTAL_EXPECTED_TESTS} (0.00%)")
            sys.exit(0)

        if results is None:
            print(f"\nScore: 0 / {TOTAL_EXPECTED_TESTS} (0.00%)")
            sys.exit(0)

        passed, total, percentage = grade(results)

        # Detailed output
        if args.verbose:
            for class_name, entry in TEST_REGISTRY.items():
                actual_lookup = dict(results.get(class_name, []))
                expected_tests = entry["tests"]

                class_passed = sum(
                    1 for t in expected_tests if actual_lookup.get(t, False)
                )
                print(f"{class_name}: {class_passed}/{len(expected_tests)}")

                for test_name in expected_tests:
                    did_pass = actual_lookup.get(test_name, False)
                    symbol = "PASS" if did_pass else "FAIL"
                    missing = "" if test_name in actual_lookup else " (missing!)"
                    print(f"  [{symbol}] {test_name}{missing}")
                print()

        # Final score
        print("=" * 50)
        print(f"Score: {passed} / {total} ({percentage:.2f}%)")
        print("=" * 50)


if __name__ == "__main__":
    main()
