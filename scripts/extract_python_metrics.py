from pathlib import Path
import subprocess
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_NAME = "pydantic" 
REPO_PATH = PROJECT_ROOT / "repos" / REPO_NAME
OUTPUT_PATH = PROJECT_ROOT / "data" / f"metrics_{REPO_NAME}.csv"

PYTHON_FILES = list(REPO_PATH.rglob("*.py"))

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "file_path",
        "loc",
        "avg_cyclomatic_complexity",
        "maintainability_index",
        "function_count"
    ])

    for py_file in PYTHON_FILES:
        rel_path = py_file.relative_to(REPO_PATH)

        # LOC
        loc_out = run_cmd(["radon", "raw", str(py_file), "-j"])
        loc = json.loads(loc_out)[str(py_file)]["loc"]

        # Cyclomatic Complexity
        cc_out = run_cmd(["radon", "cc", str(py_file), "-j"])
        cc_data = json.loads(cc_out).get(str(py_file), [])
        avg_cc = (
            sum(item["complexity"] for item in cc_data) / len(cc_data)
            if cc_data else 0
        )

        # Maintainability Index
        mi_out = run_cmd(["radon", "mi", str(py_file), "-j"])
        mi = json.loads(mi_out)[str(py_file)]["mi"]

        # Function count
        func_count = len(cc_data)

        writer.writerow([
            str(rel_path),
            loc,
            round(avg_cc, 2),
            round(mi, 2),
            func_count
        ])

print(f"Extracted metrics for {len(PYTHON_FILES)} Python files")

