"""
Generate required outputs for analysis tables from existing repo data.

Outputs (by default under output/derived/):
- selection_llm_only_top10.json/.csv
- selection_hybrid_top10.json/.csv
- cluster_assignments.csv
- file_metadata.csv
- time_series_examples.json/.csv
- summary.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "derived"

DEFAULT_TOP_K = 10
TARGET_REPOS = {"flask", "next.js"}


@dataclass(frozen=True)
class SelectionRecord:
    repo_name: str
    file_path: str
    score: float
    score_label: str


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_commit_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def detect_file_type(file_path: str) -> str:
    path = file_path.replace("\\", "/")
    lower = path.lower()
    parts = lower.split("/")
    filename = parts[-1]

    # Tests
    if (
        "tests" in parts
        or "test" in parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith(".spec.ts")
        or filename.endswith(".spec.tsx")
        or filename.endswith(".test.ts")
        or filename.endswith(".test.tsx")
        or filename.endswith(".test.js")
        or filename.endswith(".test.jsx")
    ):
        return "tests"

    # Docs
    if (
        "docs" in parts
        or filename.endswith(".md")
        or filename.endswith(".rst")
        or filename.endswith(".txt")
        or filename.endswith(".adoc")
    ):
        return "docs"

    # Dependencies / config
    dep_filenames = {
        "requirements.txt",
        "requirements.in",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "cargo.toml",
        "cargo.lock",
        "pom.xml",
        "build.gradle",
    }
    if (
        filename in dep_filenames
        or "requirements" in parts
        or filename.endswith(".lock")
        or filename.endswith(".toml")
        or filename.endswith(".yaml")
        or filename.endswith(".yml")
        or filename.endswith(".cfg")
        or filename.endswith(".ini")
    ):
        return "deps"

    return "code"


def build_commit_counts(file_changes: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    repo_file_commits: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for row in file_changes:
        repo = row.get("repo_name")
        if repo not in TARGET_REPOS:
            continue
        file_path = row.get("file_path")
        commit_hash = row.get("commit_hash")
        if not file_path or not commit_hash:
            continue
        repo_file_commits[repo][file_path].add(commit_hash)

    repo_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    for repo, files in repo_file_commits.items():
        for file_path, commits in files.items():
            repo_counts[repo][file_path] = len(commits)

    return repo_counts


def select_top_k_commit_count(
    commit_counts: Dict[str, Dict[str, int]],
    k: int = DEFAULT_TOP_K,
) -> Dict[str, List[SelectionRecord]]:
    results: Dict[str, List[SelectionRecord]] = {}
    for repo, counts in commit_counts.items():
        sorted_items = sorted(
            counts.items(),
            key=lambda x: (-x[1], x[0]),
        )[:k]
        results[repo] = [
            SelectionRecord(repo, file_path, float(count), "commit_count")
            for file_path, count in sorted_items
        ]
    return results


def select_top_k_anomaly(
    file_analysis: List[Dict],
    k: int = DEFAULT_TOP_K,
) -> Dict[str, List[SelectionRecord]]:
    repo_groups: Dict[str, List[Dict]] = defaultdict(list)
    for entry in file_analysis:
        repo = entry.get("repo_name")
        if repo in TARGET_REPOS:
            repo_groups[repo].append(entry)

    results: Dict[str, List[SelectionRecord]] = {}
    for repo, entries in repo_groups.items():
        entries_sorted = sorted(
            entries,
            key=lambda x: (-float(x.get("anomaly_score", 0.0)), x.get("file_path", "")),
        )[:k]
        results[repo] = [
            SelectionRecord(
                repo,
                e.get("file_path", ""),
                float(e.get("anomaly_score", 0.0)),
                "anomaly_score",
            )
            for e in entries_sorted
        ]
    return results


def build_cluster_assignments(file_analysis: List[Dict]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for entry in file_analysis:
        repo = entry.get("repo_name")
        if repo not in TARGET_REPOS:
            continue
        rows.append(
            {
                "repo_name": repo,
                "file_path": entry.get("file_path", ""),
                "cluster_id": str(entry.get("cluster_id", "")),
                "anomaly_score": str(entry.get("anomaly_score", "")),
            }
        )
    return rows


def build_file_metadata(
    file_analysis: List[Dict],
    file_changes: List[Dict[str, str]],
    metrics_all: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    files_by_repo: Dict[str, set] = defaultdict(set)

    for entry in file_analysis:
        repo = entry.get("repo_name")
        if repo in TARGET_REPOS:
            files_by_repo[repo].add(entry.get("file_path", ""))

    for row in file_changes:
        repo = row.get("repo_name")
        if repo in TARGET_REPOS:
            files_by_repo[repo].add(row.get("file_path", ""))

    for row in metrics_all:
        repo = row.get("repo_name")
        if repo in TARGET_REPOS:
            files_by_repo[repo].add(row.get("file_path", ""))

    rows: List[Dict[str, str]] = []
    for repo, files in files_by_repo.items():
        for file_path in sorted(f for f in files if f):
            rows.append(
                {
                    "repo_name": repo,
                    "file_path": file_path,
                    "file_type": detect_file_type(file_path),
                }
            )
    return rows


def build_commit_date_map(commit_snapshots: List[Dict]) -> Dict[Tuple[str, str], datetime]:
    commit_dates: Dict[Tuple[str, str], datetime] = {}
    for entry in commit_snapshots:
        repo = entry.get("repo_name")
        if repo not in TARGET_REPOS:
            continue
        commit_hash = entry.get("commit_hash")
        commit_date = parse_commit_date(entry.get("commit_date"))
        if not commit_hash or not commit_date:
            continue
        key = (repo, commit_hash)
        if key not in commit_dates:
            commit_dates[key] = commit_date
    return commit_dates


def build_time_series_examples(
    metrics_all: List[Dict[str, str]],
    commit_dates: Dict[Tuple[str, str], datetime],
    metric_name: str = "loc_after",
) -> Tuple[Dict[str, Dict], List[Dict[str, str]]]:
    repo_file_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in metrics_all:
        repo = row.get("repo_name")
        if repo not in TARGET_REPOS:
            continue
        file_path = row.get("file_path")
        if not file_path:
            continue
        repo_file_rows[repo][file_path].append(row)

    json_out: Dict[str, Dict] = {}
    csv_rows: List[Dict[str, str]] = []

    for repo, files in repo_file_rows.items():
        if not files:
            continue

        # Pick file with the most entries in metrics_all
        file_path = max(files.items(), key=lambda x: (len(x[1]), x[0]))[0]
        rows = files[file_path]

        series = []
        for row in rows:
            commit_hash = row.get("commit_hash")
            date = commit_dates.get((repo, commit_hash), None)
            series.append(
                {
                    "commit_hash": commit_hash or "",
                    "commit_date": date.isoformat() if date else "",
                    "metric_value": row.get(metric_name, ""),
                }
            )

        # Sort by commit_date if available, otherwise by commit_hash
        series.sort(
            key=lambda x: (
                x["commit_date"] == "",
                x["commit_date"],
                x["commit_hash"],
            )
        )

        json_out[repo] = {
            "file_path": file_path,
            "metric": metric_name,
            "series": series,
        }

        for item in series:
            csv_rows.append(
                {
                    "repo_name": repo,
                    "file_path": file_path,
                    "metric": metric_name,
                    "commit_hash": item["commit_hash"],
                    "commit_date": item["commit_date"],
                    "metric_value": item["metric_value"],
                }
            )

    return json_out, csv_rows


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_outputs(
    out_dir: Path,
    name: str,
    selections: Dict[str, List[SelectionRecord]],
) -> None:
    json_payload = {
        repo: [record.__dict__ for record in records]
        for repo, records in selections.items()
    }
    csv_rows: List[Dict[str, str]] = []
    for repo, records in selections.items():
        for record in records:
            csv_rows.append(
                {
                    "repo_name": repo,
                    "file_path": record.file_path,
                    "score": str(record.score),
                    "score_label": record.score_label,
                }
            )

    write_json(out_dir / f"{name}.json", json_payload)
    write_csv(out_dir / f"{name}.csv", csv_rows)


def main() -> None:
    ensure_out_dir(OUT_DIR)

    file_analysis = read_json(DATA_DIR / "file_analysis.json")
    file_changes = read_csv(DATA_DIR / "file_changes.csv")
    metrics_all = read_csv(DATA_DIR / "metrics_all.csv")
    commit_snapshots = read_json(DATA_DIR / "commit_snapshots.json")

    commit_counts = build_commit_counts(file_changes)
    llm_only_selections = select_top_k_commit_count(commit_counts, DEFAULT_TOP_K)
    hybrid_selections = select_top_k_anomaly(file_analysis, DEFAULT_TOP_K)

    cluster_assignments = build_cluster_assignments(file_analysis)
    file_metadata = build_file_metadata(file_analysis, file_changes, metrics_all)

    commit_dates = build_commit_date_map(commit_snapshots)
    time_series_json, time_series_csv = build_time_series_examples(
        metrics_all, commit_dates, metric_name="loc_after"
    )

    write_selection_outputs(OUT_DIR, "selection_llm_only_top10", llm_only_selections)
    write_selection_outputs(OUT_DIR, "selection_hybrid_top10", hybrid_selections)

    write_csv(OUT_DIR / "cluster_assignments.csv", cluster_assignments)
    write_csv(OUT_DIR / "file_metadata.csv", file_metadata)
    write_json(OUT_DIR / "time_series_examples.json", time_series_json)
    write_csv(OUT_DIR / "time_series_examples.csv", time_series_csv)

    summary = {
        "repos": sorted(TARGET_REPOS),
        "top_k": DEFAULT_TOP_K,
        "outputs": [
            "selection_llm_only_top10.json",
            "selection_llm_only_top10.csv",
            "selection_hybrid_top10.json",
            "selection_hybrid_top10.csv",
            "cluster_assignments.csv",
            "file_metadata.csv",
            "time_series_examples.json",
            "time_series_examples.csv",
        ],
        "counts": {
            "file_analysis_entries": len(file_analysis),
            "file_changes_rows": len(file_changes),
            "metrics_all_rows": len(metrics_all),
        },
    }
    write_json(OUT_DIR / "summary.json", summary)

    print("Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
