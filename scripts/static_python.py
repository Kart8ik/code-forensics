import ast
import json
import os
import subprocess
import sys

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.returncode, result.stderr

def analyze_python_file(path):
    metrics = {
        "loc": 0,
        "cc": 0,
        "mi": None,
        "function_count": 0,
        "import_count": 0
    }

    # Use sys.executable to run radon as a module (works in virtual envs)
    python = sys.executable

    try:
        # Check if file exists before analyzing
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return None

        # LOC
        stdout, returncode, stderr = run([python, "-m", "radon", "raw", path, "-j"])
        if returncode != 0 or not stdout.strip():
            print(f"radon raw failed for {path}: {stderr.strip()}")
            return None
        
        raw = json.loads(stdout)
        # Radon may return path with forward slashes, normalize for lookup
        normalized_path = path.replace("\\", "/")
        file_key = None
        for key in raw.keys():
            if key == path or key == normalized_path or key.replace("\\", "/") == normalized_path:
                file_key = key
                break
        
        if file_key is None:
            print(f"Path {path} not found in radon raw output (keys: {list(raw.keys())})")
            return None
        metrics["loc"] = raw[file_key].get("loc", 0)

        # MI (Maintainability Index)
        stdout, returncode, stderr = run([python, "-m", "radon", "mi", path, "-j"])
        if returncode == 0 and stdout.strip():
            mi = json.loads(stdout)
            for key in mi.keys():
                if key == path or key.replace("\\", "/") == normalized_path:
                    if "mi" in mi[key]:
                        metrics["mi"] = round(mi[key]["mi"], 2)
                    break

        # Cyclomatic Complexity
        stdout, returncode, stderr = run([python, "-m", "radon", "cc", path, "-j"])
        if returncode == 0 and stdout.strip():
            cc_json = json.loads(stdout)
            cc_data = None
            for key in cc_json.keys():
                if key == path or key.replace("\\", "/") == normalized_path:
                    cc_data = cc_json[key]
                    break
            if cc_data:
                metrics["cc"] = round(
                    sum(item["complexity"] for item in cc_data) / len(cc_data), 2
                )
                metrics["function_count"] = len(cc_data)

        # AST for imports
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read())
            metrics["import_count"] = sum(
                isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)
            )

    except json.JSONDecodeError as e:
        print(f"JSON parse error for {path}: {e}")
        return None
    except SyntaxError as e:
        print(f"Python syntax error in {path}: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing {path}: {e}")
        return None

    return metrics
