"""Quick analysis of ML pipeline output."""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load data
with open(DATA_DIR / "file_analysis.json") as f:
    data = json.load(f)

metrics_df = pd.read_csv(DATA_DIR / "metrics_all.csv")

print(f"Total files analyzed: {len(data)}")
print()

# By repo
repos = Counter(d["repo_name"] for d in data)
print("Files by repository:")
for repo, count in repos.items():
    print(f"  {repo}: {count}")
print()

# Top anomalies
print("Top 15 by anomaly score:")
sorted_data = sorted(data, key=lambda x: x["anomaly_score"], reverse=True)[:15]
for d in sorted_data:
    print(f"  {d['anomaly_score']:.4f} | {d['repo_name']}/{d['file_path'][:50]:50} | n={d['n_commits']:3} | churn={d['churn_label']}")
print()

# High churn files
print("High churn files (sample):")
high_churn = [d for d in data if d["churn_label"] == "high"][:10]
for d in high_churn:
    print(f"  {d['repo_name']}/{d['file_path'][:50]:50} | n={d['n_commits']:3} | rate={d['raw_features']['churn_rate']:.2f}")
print()

# Files with increasing complexity trend
print("Files with INCREASING complexity trend:")
increasing = [d for d in data if d["historical_trends"]["cc"]["label"] == "increasing"]
for d in increasing[:10]:
    print(f"  {d['repo_name']}/{d['file_path'][:50]:50} | cc_trend={d['historical_trends']['cc']['value']:.4f}")
print(f"  Total: {len(increasing)}")
print()

# Files with decreasing MI (maintainability)
print("Files with DECREASING maintainability index:")
decreasing_mi = [d for d in data if d["historical_trends"]["mi"]["label"] == "decreasing"]
for d in decreasing_mi[:10]:
    print(f"  {d['repo_name']}/{d['file_path'][:50]:50} | mi_trend={d['historical_trends']['mi']['value']:.4f}")
print(f"  Total: {len(decreasing_mi)}")
print()

# Sample of oscillating volatility
print("Files with oscillating imports volatility:")
oscillating = [d for d in data if d["imports_volatility_label"] == "oscillating"]
for d in oscillating[:5]:
    print(f"  {d['repo_name']}/{d['file_path'][:50]:50} | std_imports={d['raw_features']['std_imports_after']:.2f}")
print(f"  Total: {len(oscillating)}")

# Verify specific files against raw metrics
print("\n" + "="*70)
print("VERIFICATION: Cross-checking with raw metrics")
print("="*70)

# Check flask/app.py
flask_app = metrics_df[(metrics_df['repo_name']=='flask') & (metrics_df['file_path']=='src/flask/app.py')]
print(f"\nFlask app.py: {len(flask_app)} commits")
print(f"  LOC delta: mean={flask_app['loc_delta'].abs().mean():.2f}, non-zero={((flask_app['loc_delta'] != 0).sum())}/{len(flask_app)}")
print(f"  CC after:  mean={flask_app['cc_after'].mean():.2f}, std={flask_app['cc_after'].std():.4f}")
print(f"  Imports:   mean={flask_app['imports_after'].mean():.2f}, std={flask_app['imports_after'].std():.2f}")

# Check flask __init__.py (high anomaly)
flask_init = metrics_df[(metrics_df['repo_name']=='flask') & (metrics_df['file_path']=='src/flask/__init__.py')]
print(f"\nFlask __init__.py: {len(flask_init)} commits (HIGHEST ANOMALY)")
print(f"  LOC delta: mean_abs={flask_init['loc_delta'].abs().mean():.2f}")
print(f"  CC after:  min={flask_init['cc_after'].min():.2f}, max={flask_init['cc_after'].max():.2f}, std={flask_init['cc_after'].std():.4f}")
print(f"  MI after:  min={flask_init['mi_after'].min():.2f}, max={flask_init['mi_after'].max():.2f}")
print("  -> Extreme oscillation between cc=0/3 and mi=54/100 explains high anomaly score!")

# Check a next.js router file
nextjs_router = metrics_df[metrics_df['file_path'].str.contains('router-reducer', na=False)]
if len(nextjs_router) > 0:
    print(f"\nNext.js router-reducer files: {len(nextjs_router)} metric rows")
    print(f"  Unique files: {nextjs_router['file_path'].nunique()}")
    print(f"  Mean CC after: {nextjs_router['cc_after'].mean():.2f}")
    print(f"  Mean imports: {nextjs_router['imports_after'].mean():.2f}")

# Sanity check: churn rate calculation
print("\n" + "="*70)
print("SANITY CHECK: Churn rate calculation")
print("="*70)
for file_record in [d for d in data if d['file_path'] == 'src/flask/app.py'][:1]:
    file_df = metrics_df[(metrics_df['repo_name']==file_record['repo_name']) & 
                         (metrics_df['file_path']==file_record['file_path'])]
    actual_churn = (file_df['loc_delta'] != 0).sum() / len(file_df)
    print(f"Flask app.py:")
    print(f"  Reported churn_rate: {file_record['raw_features']['churn_rate']:.4f}")
    print(f"  Calculated from raw: {actual_churn:.4f}")
    print(f"  Match: {'YES' if abs(actual_churn - file_record['raw_features']['churn_rate']) < 0.01 else 'NO'}")
