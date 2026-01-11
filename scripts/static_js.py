# scripts/static_js.py
"""
JavaScript/TypeScript static analysis using Babel AST parser.
Calls a Node.js script that uses @babel/parser for accurate parsing.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# Path to the Node.js analyzer script
SCRIPT_DIR = Path(__file__).resolve().parent
AST_ANALYZER = SCRIPT_DIR / "analyze_js_ast.js"


def analyze_js_file(path):
    """Analyze a JavaScript/TypeScript file using Babel AST parser."""
    metrics = {
        "loc": 0,
        "cc": 0,
        "mi": None,
        "function_count": 0,
        "import_count": 0
    }

    # Check if file exists
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return None

    # Check if Node.js analyzer script exists
    if not AST_ANALYZER.exists():
        print(f"AST analyzer script not found: {AST_ANALYZER}")
        return None

    try:
        # Call the Node.js script
        result = subprocess.run(
            ["node", str(AST_ANALYZER), path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(SCRIPT_DIR)  # Run from scripts dir so node_modules is found
        )

        if result.returncode != 0:
            # Try to parse error from stderr
            try:
                err_data = json.loads(result.stderr.strip())
                print(f"AST analysis failed for {path}: {err_data.get('error', 'Unknown error')}")
            except json.JSONDecodeError:
                print(f"AST analysis failed for {path}: {result.stderr.strip()}")
            return None

        if not result.stdout.strip():
            print(f"No output from AST analyzer for {path}")
            return None

        # Parse the JSON output
        data = json.loads(result.stdout.strip())

        if "error" in data:
            print(f"AST analysis error for {path}: {data['error']}")
            return None

        metrics["loc"] = data.get("loc", 0)
        metrics["cc"] = data.get("cc", 0)
        metrics["mi"] = data.get("mi")  # Will be None
        metrics["function_count"] = data.get("function_count", 0)
        metrics["import_count"] = data.get("import_count", 0)

    except subprocess.TimeoutExpired:
        print(f"AST analysis timed out for {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {path}: {e}")
        return None
    except FileNotFoundError:
        print("Node.js not found. Please install Node.js to analyze JS/TS files.")
        return None
    except Exception as e:
        print(f"Error analyzing {path}: {e}")
        return None

    return metrics
