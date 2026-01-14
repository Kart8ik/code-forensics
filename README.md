# Code Forensics

## Objective

To conduct an **empirical study** evaluating whether combining **static code metrics**, **classical machine learning**, and **LLM-based semantic reasoning** provides better code quality analysis than using each technique independently.

## Repository Structure

```
code-forensics/
├── scripts/
│   ├── extract_git_data.py       # Git history extraction
│   ├── run_static_analysis.py    # Main static analysis runner
│   ├── static_python.py          # Python metrics (radon)
│   ├── static_js.py              # JS/TS metrics (Babel AST)
│   ├── analyze_js_ast.js         # Node.js Babel AST analyzer
│   ├── ml_pipeline.py            # ML layer: clustering + anomaly detection
│   ├── package.json              # Node.js dependencies
│   └── extract_python_metrics.py # old version
├── data/
│   ├── commits.csv               # Commit metadata
│   ├── file_changes.csv          # Files changed per commit
│   ├── diffs.csv                 # Code diffs
│   ├── metrics_all.csv           # Static analysis metrics
│   ├── file_analysis.json        # ML output: per-file analysis
│   └── cluster_summary.json      # ML output: cluster summaries
├── repos/
│   └── <cloned_repos>/           # Target repositories
└── cf-env/                       # Python virtual environment
```

## Data Outputs

| File | Description |
|------|-------------|
| `commits.csv` | Commit hash, author, date, message |
| `file_changes.csv` | repo_name, commit_hash, file_path, change_type |
| `diffs.csv` | Code diffs for source files |
| `metrics_all.csv` | Static metrics: LOC, cyclomatic complexity, function count, imports |
| `file_analysis.json` | ML output: per-file features, cluster assignment, anomaly scores, labels |
| `cluster_summary.json` | ML output: aggregate stats and labels for each cluster |

## Setup

### 1. Python Environment
```powershell
# Activate virtual environment
.\cf-env\Scripts\Activate.ps1    # PowerShell

# Or install dependencies manually
pip install -r requirements.txt
```

### 2. Node.js Dependencies (for JS/TS analysis)
```powershell
cd scripts
npm install
```

### 3. Clone Target Repositories
```powershell
cd repos
git clone <repo_url>
```

## Running

### Step 1: Extract Git Data
```powershell
python scripts/extract_git_data.py
```
- Edit `REPO_NAME` in the script to match your repo folder
- Extracts commits, file changes, and diffs to `data/`

### Step 2: Run Static Analysis
```powershell
python scripts/run_static_analysis.py
```
- Analyzes Python files using `radon` (LOC, complexity, maintainability index)
- Analyzes JS/TS files using Babel AST parser (LOC, complexity, functions, imports)
- Outputs to `data/metrics_all.csv`

### Step 3: Run ML Pipeline
```powershell
python scripts/ml_pipeline.py
```
- Groups metrics by file, computes temporal features (trends, volatility, churn)
- Runs KMeans clustering to find files with similar evolution patterns
- Runs Isolation Forest to flag anomalous files
- Applies rule-based labels (high/low churn, increasing/stable complexity, etc.)
- Outputs to `data/file_analysis.json` and `data/cluster_summary.json`

Optional parameters:
```powershell
python scripts/ml_pipeline.py --min-commits 10 --recent-window 15
```
- `--min-commits`: Skip files with fewer than N commits (default: 5)
- `--recent-window`: How many recent commits to use for "recent" trends (default: 20)

## Extraction Rules

- Processes at most `MAX_COMMITS` (default 500) from HEAD backwards
- Skips merge commits (`len(commit.parents) != 1`)
- Writes diffs only for files with `SOURCE_EXTENSIONS` (`.py`, `.js`, `.ts`, `.tsx`, `.jsx`)
- Diff text is capped at `MAX_DIFF_CHARS` to keep rows compact
- CSVs are initialized with headers; subsequent runs append

## Static Analysis Metrics

| Metric | Python | JS/TS |
|--------|--------|-------|
| LOC | Lines of code (radon) | Lines of code (AST) |
| CC | Cyclomatic complexity (radon) | Decision points (AST) |
| MI | Maintainability index (radon) | N/A |
| Functions | Function count | Functions, methods, arrows |
| Imports | Import statements | ES6 imports + require() |

## ML Pipeline Features

The ML pipeline computes 20 features per file to capture how it evolves over time:

| Category | Features | What it tells you |
|----------|----------|-------------------|
| Change Intensity | churn_rate, mean_abs_loc/cc/imports/func_delta | How actively is this file being modified? |
| Volatility | std_cc_after, std_mi_after, std_imports_after | Are the metrics stable or bouncing around? |
| Trends | cc_trend, mi_trend, imports_trend | Is complexity growing? Maintainability dropping? |
| Structural Load | mean_cc_after, mean_mi_after, mean_imports_after, mean_func_after | How complex is this file in absolute terms? |
| Recent Window | recent_*_trend, recent_*_volatility | Same questions but just for the last N commits |

### ML Outputs

**file_analysis.json** - One entry per file:
- `cluster_id`: Which cluster this file belongs to
- `anomaly_score`: 0-1 score (higher = weirder evolution pattern)
- `churn_label`: low / moderate / high
- `historical_trends`: cc, mi, imports with values and labels
- `recent_trends`: Same but for recent commits, plus vs_historical comparison
- `raw_features`: The actual numeric values

**cluster_summary.json** - One entry per cluster:
- `size`: Number of files in cluster
- `avg_*`: Average feature values
- `derived_cluster_label`: Human-readable description (e.g., "high churn, stable complexity")

## Notes

- The static analysis is optimized to group files by commit, minimizing git checkouts
- For large repos like Next.js (~6k files), expect ~1-2 hours for full analysis
- Keep repos up to date (`git pull`) before extracting
- For analysis, load CSVs with pandas or DuckDB
- The ML pipeline produces JSON outputs designed for downstream LLM consumption
- Anomaly detection uses Isolation Forest - high scores mean unusual evolution patterns
- Clustering uses KMeans with automatic k selection via silhouette score