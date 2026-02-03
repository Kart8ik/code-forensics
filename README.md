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
│   ├── ml_pipeline.py            # ML layer (features, clustering, anomalies)
│   ├── faiss_rag.py              # Diff-only FAISS RAG indexer/retriever
│   ├── llm_pipeline_rag.py       # LLM interpretation (file + repo synthesis)
│   ├── analyze_output.py         # Output analysis helpers
│   ├── extract_python_metrics.py # Old version (legacy)
│   ├── package.json              # Node.js dependencies
│   └── package-lock.json
├── data/
│   ├── commits.csv               # Commit metadata
│   ├── file_changes.csv          # Files changed per commit
│   ├── diffs.csv                 # Code diffs
│   ├── metrics_all.csv           # Static analysis metrics
│   ├── file_analysis.json        # ML per-file outputs
│   ├── cluster_summary.json      # ML cluster summaries
│   ├── faiss.index               # FAISS index (diff embeddings)
│   └── faiss_documents.pkl       # FAISS document store
├── output/
│   ├── file_explanations.json    # LLM file-level explanations
│   ├── repository_report.md      # LLM repo synthesis (markdown)
│   ├── repository_report.json    # LLM repo synthesis (structured)
│   ├── grounding_metrics.json    # LLM compliance metrics
│   └── checkpoint.json           # Intermediate LLM output
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
| `file_analysis.json` | ML per-file features, labels, anomaly scores |
| `cluster_summary.json` | ML cluster summaries and labels |
| `faiss.index` | FAISS vector index over diffs |
| `faiss_documents.pkl` | FAISS document/metadata store |
| `file_explanations.json` | LLM file-level explanations |
| `repository_report.md` | LLM repo-level synthesis (markdown) |
| `repository_report.json` | LLM repo-level synthesis (structured) |
| `grounding_metrics.json` | LLM compliance/grounding metrics |

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
- Edit `REPO_NAME` in `scripts/extract_git_data.py` to match your repo folder
- Extracts commits, file changes, and diffs to `data/`

### Step 2: Run Static Analysis
```powershell
python scripts/run_static_analysis.py
```
- Analyzes Python files using `radon` (LOC, complexity, maintainability index)
- Analyzes JS/TS files using Babel AST parser (LOC, complexity, functions, imports)
- Outputs to `data/metrics_all.csv`

### Step 3: Run ML Layer (per-file modeling)
```powershell
python scripts/ml_pipeline.py --repo <repo_name>
```
- Filters modeling to files that exist at repository HEAD
- Builds per-file time series, clustering, and anomaly scores
- Outputs `data/file_analysis.json` and `data/cluster_summary.json`

### Step 4: Build FAISS Diff Index (optional but recommended)
```powershell
python scripts/faiss_rag.py
```
- Builds a diff-only FAISS index from `data/diffs.csv`
- Outputs `data/faiss.index` and `data/faiss_documents.pkl`

### Step 5: Run LLM Interpretation
```powershell
python scripts/llm_pipeline_rag.py
```
- Requires `GROQ_API_KEY` in environment or `.env`
- Produces file-level explanations and repo synthesis in `output/`

## Extraction Rules

- Processes at most `MAX_COMMITS` (default 500) from HEAD backwards
- Skips merge commits (`len(commit.parents) != 1`)
- Writes diffs only for files with `SOURCE_EXTENSIONS` (`.py`, `.js`, `.ts`, `.tsx`, `.jsx`)
- Diff text is capped at `MAX_DIFF_CHARS` to keep rows compact
- CSVs are initialized with headers; subsequent runs append

## ML Layer Notes

- Files are modeled only if they exist at repository HEAD
- File identity is defined by `file_path` (renames appear as delete + add)
- Filtering happens at grouping stage; historical metrics are retained

## Static Analysis Metrics

| Metric | Python | JS/TS |
|--------|--------|-------|
| LOC | Lines of code (radon) | Lines of code (AST) |
| CC | Cyclomatic complexity (radon) | Decision points (AST) |
| MI | Maintainability index (radon) | N/A |
| Functions | Function count | Functions, methods, arrows |
| Imports | Import statements | ES6 imports + require() |

## Notes

- The static analysis is optimized to group files by commit, minimizing git checkouts
- For large repos like Next.js (~6k files), expect ~1-2 hours for full analysis
- Keep repos up to date (`git pull`) before extracting
- For analysis, load CSVs with pandas or DuckDB
