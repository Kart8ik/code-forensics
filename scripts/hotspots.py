import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_REPO = "requests"   # change to requests / core

commits = pd.read_csv(DATA_DIR / "commits.csv")
files = pd.read_csv(DATA_DIR / "file_changes.csv")
metrics = pd.read_csv(DATA_DIR / "metrics_all.csv")

files["file_path"] = files["file_path"].astype(str)
metrics["file_path"] = metrics["file_path"].astype(str)

# ---- churn ----
churn = (
    files.groupby("file_path")
    .size()
    .reset_index(name="change_count")
)

# ---- merge ----
merged = churn.merge(metrics, on="file_path", how="inner")

# ---- filter noise ----
exclude_patterns = (
    "setup.py",
    "__init__.py",
    "tests/",
    ".github/",
    "Cargo",
)

merged = merged[
    ~merged["file_path"].str.contains("|".join(exclude_patterns), na=False)
]

# ---- risk score ----
merged["risk_score"] = (
    merged["change_count"] * merged["cc_delta"].abs()
)

# ---- repo-specific ----
hotspots = merged[merged["repo_name"] == TARGET_REPO]

# ---- sort ----
hotspots = hotspots.sort_values(
    by="risk_score", ascending=False
)

# ---- append ----
output = DATA_DIR / "hotspots.csv"
hotspots.to_csv(
    output,
    mode="a",
    header=not output.exists(),
    index=False
)

print(f"Hotspots appended for repo: {TARGET_REPO}")
print("\nTop 10 risky files:")
print(hotspots.head(10)[
    ["repo_name", "file_path", "change_count", "cc_after", "mi_after", "risk_score"]
])
