# Code Forensics

## Objective

To conduct an **empirical study** evaluating whether combining **static code metrics**, **classical machine learning**, and **LLM-based semantic reasoning** provides better code quality analysis than using each technique independently.

## What’s here

code-forensics/
 ├── scripts/
 │    └── extract_git_data.py
 ├── data/    `make this if needed even though it is made automatically in the script`
 │    ├── commits.csv
 │    ├── file_changes.csv
 │    └── diffs.csv
 └── repos/   `add the cloned repos that are gonna be worked on here`
      ├── repo_1/
      └── repo_2/


## Extraction rules (current)
- Processes at most `MAX_COMMITS` (default 500) from HEAD backwards. (latest 500 commits)
- Skips merge commits (`len(commit.parents) != 1`).
- Writes diffs only for files ending with `SOURCE_EXTENSIONS` (default: `.py`, `.java`) and only for `A`/`M` changes.
- Diff text is capped at `MAX_DIFF_CHARS` to keep rows compact.
- CSVs are initialized with headers if missing; subsequent runs append.

## Setup
1) Activate the venv (optional but recommended):
   - PowerShell: `.\cf-env\Scripts\Activate.ps1`
   - CMD: `.\cf-env\Scripts\activate.bat`
2) Ensure dependencies if not using the bundled venv: `pip install -r requirements.txt`
3) Clone the target repo into `repos/<repo_1>` (e.g., `repos/flask`).

## Running the extractor
1) Open `scripts/extract_git_data.py` and set `REPO_NAME` to the folder name under `repos/`.
2) (Optional) Adjust `MAX_COMMITS`, `SOURCE_EXTENSIONS`, or `MAX_DIFF_CHARS`.
3) Run:
   ```
   python scripts/extract_git_data.py
   ```
4) Outputs land in `data/` (created if absent). Re-running appends; delete/rename existing CSVs if you need a fresh run.

## Notes & next steps
- Keep repos up to date (`git pull`) before extracting to capture recent history.
- For larger analyses, load the CSVs with pandas or DuckDB.

