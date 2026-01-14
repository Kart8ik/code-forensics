from pathlib import Path
import csv
from collections import defaultdict
from git import Repo
from static_python import analyze_python_file
from static_js import analyze_js_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPOS_DIR = PROJECT_ROOT / "repos"

METRICS_OUT = DATA_DIR / "metrics_all.csv"

SOURCE_EXTS = {
    "python": (".py",),
    "js": (".js", ".jsx", ".ts", ".tsx")
}

# Cache for Repo objects per repo_name
_repo_cache = {}
_current_checkout = {}  # Track current checkout per repo

def get_repo(repo_name):
    """Get or create a Repo object for the given repo_name."""
    if repo_name not in _repo_cache:
        repo_path = REPOS_DIR / repo_name
        repo = Repo(repo_path)
        # Enable long paths on Windows to handle repos like next.js with very long filenames
        try:
            with repo.config_writer() as cw:
                cw.set_value("core", "longpaths", "true")
        except Exception:
            pass  # Ignore if we can't set config
        _repo_cache[repo_name] = repo
    return _repo_cache[repo_name]

def git_checkout(repo_name, commit):
    """Checkout a specific commit in the target repository (with caching)."""
    # Skip if already at this commit
    if _current_checkout.get(repo_name) == commit:
        return True
    
    repo = get_repo(repo_name)
    try:
        repo.git.checkout(commit, force=True, quiet=True)
        _current_checkout[repo_name] = commit
        return True
    except Exception as e:
        print(f"Checkout failed for {repo_name}@{commit}: {e}")
        return False

def analyze_file(path):
    if path.endswith(SOURCE_EXTS["python"]):
        return "python", analyze_python_file(path)
    if path.endswith(SOURCE_EXTS["js"]):
        return "js", analyze_js_file(path)
    return None, None

# Load and group file changes by (repo_name, commit_hash)
print("Loading and grouping file changes...")
commit_files = defaultdict(list)  # (repo_name, commit) -> [file_paths]

with open(DATA_DIR / "file_changes.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["repo_name"], row["commit_hash"])
        commit_files[key].append(row["file_path"])

total_commits = len(commit_files)
total_files = sum(len(files) for files in commit_files.values())
print(f"Found {total_files} files across {total_commits} commits")

# Process commits in order
with open(METRICS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "repo_name", "commit_hash", "file_path", "language",
        "loc_before", "loc_after", "loc_delta",
        "cc_before", "cc_after", "cc_delta",
        "mi_before", "mi_after", "mi_delta",
        "func_before", "func_after", "func_delta",
        "imports_before", "imports_after", "imports_delta"
    ])

    processed_commits = 0
    processed_files = 0
    
    for (repo_name, commit), file_paths in commit_files.items():
        processed_commits += 1
        
        # Progress update every 10 commits
        if processed_commits % 10 == 0:
            print(f"Progress: {processed_commits}/{total_commits} commits ({processed_files} files analyzed)")
        
        # Step 1: Checkout parent commit (commit~1) and analyze all files
        parent_commit = f"{commit}~1"
        if not git_checkout(repo_name, parent_commit):
            print(f"Skipping commit {commit}: cannot checkout parent")
            continue
        
        # Analyze all files at parent commit
        before_metrics = {}
        for file_path in file_paths:
            full_path = REPOS_DIR / repo_name / file_path
            if not full_path.exists():
                # File didn't exist before (newly added in this commit)
                continue
            
            lang, metrics = analyze_file(str(full_path))
            if metrics:
                before_metrics[file_path] = (lang, metrics)
        
        # Step 2: Checkout the actual commit and analyze files that existed before
        if not git_checkout(repo_name, commit):
            print(f"Skipping commit {commit}: cannot checkout")
            continue
        
        # Analyze files and write results
        for file_path, (lang, before) in before_metrics.items():
            full_path = REPOS_DIR / repo_name / file_path
            
            if not full_path.exists():
                # File was deleted in this commit
                continue
            
            _, after = analyze_file(str(full_path))
            if not after:
                continue
            
            processed_files += 1
            
            writer.writerow([
                repo_name, commit, file_path, lang,
                before["loc"], after["loc"], after["loc"] - before["loc"],
                before["cc"], after["cc"], after["cc"] - before["cc"],
                before["mi"], after["mi"],
                None if before["mi"] is None or after["mi"] is None else after["mi"] - before["mi"],
                before["function_count"], after["function_count"],
                after["function_count"] - before["function_count"],
                before["import_count"], after["import_count"],
                after["import_count"] - before["import_count"]
            ])

print(f"\nStatic analysis complete!")
print(f"Processed {processed_commits} commits, {processed_files} files")
print(f"Results saved to {METRICS_OUT}")
