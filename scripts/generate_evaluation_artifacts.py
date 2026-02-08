"""
Generate evaluation artifacts (3 tables + 1 figure) from existing outputs.

Inputs (from output/derived/):
- selection_llm_only_top10.json
- selection_hybrid_top10.json
- cluster_assignments.csv
- file_metadata.csv
- time_series_examples.json

Outputs (to output/eval/):
- table_ii_selection_quality.csv
- table_iii_cluster_coverage.csv
- table_iv_file_type_distribution.csv
- time_series_example.pdf
- time_series_example.png
- summary.json

Deterministic, reproducible, no external dependencies beyond pandas/matplotlib.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
IN_DIR = BASE_DIR / "output" / "derived"
OUT_DIR = BASE_DIR / "output" / "eval"

K_FIXED = 10


@dataclass(frozen=True)
class SelectionRow:
    repo: str
    file_path: str
    mode: str


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_selection(path: Path, mode: str) -> pd.DataFrame:
    data = read_json(path)
    rows: List[Dict[str, str]] = []
    for repo, files in data.items():
        for entry in files:
            file_path = entry.get("file_path") if isinstance(entry, dict) else str(entry)
            if not file_path:
                continue
            rows.append(SelectionRow(repo=repo, file_path=file_path, mode=mode).__dict__)
    return pd.DataFrame(rows)


def build_table_ii(sel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_norm = meta.rename(columns={"repo_name": "repo"})
    merged = sel.merge(meta_norm, on=["repo", "file_path"], how="left")
    counts = (
        merged.groupby(["repo", "mode", "file_type"]).size().reset_index(name="count")
    )
    dominant = (
        counts.sort_values(["repo", "mode", "count", "file_type"], ascending=[True, True, False, True])
        .groupby(["repo", "mode"], as_index=False)
        .first()
    )
    dominant = dominant.rename(columns={"file_type": "dominant_file_type"})
    dominant["k_fixed"] = K_FIXED
    return dominant[["repo", "mode", "k_fixed", "dominant_file_type", "count"]]


def build_table_iii(llm_sel: pd.DataFrame, hyb_sel: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    def coverage(sel: pd.DataFrame, mode: str) -> pd.DataFrame:
        clusters_norm = clusters.rename(columns={"repo_name": "repo"})
        merged = sel.merge(clusters_norm, on=["repo", "file_path"], how="left")
        out = (
            merged.groupby("repo")["cluster_id"].nunique().reset_index(name="clusters_covered")
        )
        out["mode"] = mode
        out["files_explained"] = K_FIXED
        return out[["repo", "mode", "files_explained", "clusters_covered"]]

    cov_llm = coverage(llm_sel, "LLM-only")
    cov_hyb = coverage(hyb_sel, "Hybrid")
    return pd.concat([cov_llm, cov_hyb], ignore_index=True)


def build_table_iv(sel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_norm = meta.rename(columns={"repo_name": "repo"})
    merged = sel.merge(meta_norm, on=["repo", "file_path"], how="left")
    counts = (
        merged.groupby(["repo", "mode", "file_type"]).size().reset_index(name="count")
    )
    counts["percent"] = (
        counts.groupby(["repo", "mode"])["count"].transform(lambda x: 100 * x / x.sum())
    )
    dist = counts[["repo", "mode", "file_type", "percent"]].copy()
    dist["percent"] = dist["percent"].round(2)
    return dist


def build_time_series_plot(ts_data: Dict) -> Dict[str, str]:
    # Deterministic selection: alphabetical by repo name
    if not ts_data:
        raise ValueError("time_series_examples.json is empty")

    repo_name = sorted(ts_data.keys())[0]
    entry = ts_data[repo_name]
    series = entry.get("series", [])
    if not series:
        raise ValueError(f"No series data for repo: {repo_name}")

    df = pd.DataFrame(series)
    df["commit_date"] = pd.to_datetime(df["commit_date"], errors="coerce", utc=True)
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df.dropna(subset=["commit_date", "metric_value"]).copy()
    df["commit_date"] = df["commit_date"].dt.tz_convert(None)
    df = df.sort_values(["commit_date", "commit_hash"], na_position="last")

    metric = entry.get("metric", "loc_after")

    plt.figure(figsize=(6.0, 3.5))
    plt.plot(df["commit_date"], df["metric_value"])
    plt.xlabel("Commit Date")
    plt.ylabel(metric)
    plt.title(f"{repo_name}: {entry.get('file_path', '')}")
    plt.tight_layout()

    pdf_path = OUT_DIR / "time_series_example.pdf"
    png_path = OUT_DIR / "time_series_example.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close()

    return {
        "repo": repo_name,
        "file_path": entry.get("file_path", ""),
        "metric": metric,
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def main() -> None:
    ensure_out_dir()

    meta = pd.read_csv(IN_DIR / "file_metadata.csv")
    clusters = pd.read_csv(IN_DIR / "cluster_assignments.csv")

    llm_sel = load_selection(IN_DIR / "selection_llm_only_top10.json", "LLM-only")
    hyb_sel = load_selection(IN_DIR / "selection_hybrid_top10.json", "Hybrid")

    sel_all = pd.concat([llm_sel, hyb_sel], ignore_index=True)

    table_ii = build_table_ii(sel_all, meta)
    table_iii = build_table_iii(llm_sel, hyb_sel, clusters)
    table_iv = build_table_iv(sel_all, meta)

    table_ii.to_csv(OUT_DIR / "table_ii_selection_quality.csv", index=False)
    table_iii.to_csv(OUT_DIR / "table_iii_cluster_coverage.csv", index=False)
    table_iv.to_csv(OUT_DIR / "table_iv_file_type_distribution.csv", index=False)

    ts_data = read_json(IN_DIR / "time_series_examples.json")
    figure_info = build_time_series_plot(ts_data)

    summary = {
        "inputs_dir": str(IN_DIR),
        "outputs_dir": str(OUT_DIR),
        "k_fixed": K_FIXED,
        "tables": {
            "table_ii_selection_quality": "table_ii_selection_quality.csv",
            "table_iii_cluster_coverage": "table_iii_cluster_coverage.csv",
            "table_iv_file_type_distribution": "table_iv_file_type_distribution.csv",
        },
        "figure": figure_info,
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Artifacts written to:", OUT_DIR)


if __name__ == "__main__":
    main()
