"""
ML Pipeline for Temporal Code Evolution Analysis

Basically, this takes all those raw commit metrics and turns them into something
actually useful - per-file feature vectors that capture how each file has evolved
over time.

What it does:
1. Groups all the commit data by file (not repo - we care about individual files)
2. Computes a bunch of features: churn, trends, volatility, etc.
3. Runs clustering to find files that behave similarly
4. Runs anomaly detection to flag weird evolution patterns
5. Spits out JSON that an LLM can later explain in plain English

The key insight here is that we're NOT predicting anything - just describing
patterns. The LLM layer handles the "so what does this mean" part.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


# =============================================================================
# CONFIG - tweak these if the defaults don't work for your data
# =============================================================================

# Skip files with too few commits - need enough history to see patterns
MIN_COMMITS_DEFAULT = 5

# How many recent commits to look at for "recent" trends
# 20 is a decent window, but bump it up if your repos have lots of commits
RECENT_WINDOW_DEFAULT = 20

# KMeans will try k=3 through k=10 and pick the best one via silhouette score
K_MIN = 3
K_MAX = 10

# What fraction of files do we expect to be "weird"? 10% seems reasonable
ISOLATION_CONTAMINATION = 0.1

# For reproducibility - same seed = same results every time
RANDOM_STATE = 42

# =============================================================================
# THRESHOLDS FOR LABELING - these turn numbers into words like "high"/"low"
# =============================================================================

# Churn: what % of commits actually change the file?
# <= 30% = low, 30-70% = moderate, > 70% = high (file changes a lot)
CHURN_THRESHOLDS = (0.3, 0.7)

# How much does CC need to change per commit to count as "increasing"?
# 0.01 is pretty sensitive - even small slopes get flagged
CC_TREND_THRESHOLD = 0.01

# Same idea for maintainability index - 0.1 per commit is noticeable
MI_TREND_THRESHOLD = 0.1

# And for imports - 0.05 new imports per commit on average
IMPORTS_TREND_THRESHOLD = 0.05

# When is a metric "oscillating" vs "stable"?
# If std dev > 1.0, there's real movement happening
VOLATILITY_THRESHOLD = 1.0

# For comparing recent vs historical: how different is "different"?
# 20% change = worth noting
RECENT_COMPARISON_THRESHOLD = 0.2


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METRICS_FILE = DATA_DIR / "metrics_all.csv"
COMMITS_FILE = DATA_DIR / "commits.csv"
OUTPUT_FILE_ANALYSIS = DATA_DIR / "file_analysis.json"
OUTPUT_CLUSTER_SUMMARY = DATA_DIR / "cluster_summary.json"


# =============================================================================
# THE 19 FEATURES WE COMPUTE PER FILE
# =============================================================================

FEATURE_NAMES = [
    # How much does this file get touched? (5 features)
    "churn_rate",              # % of commits that actually change LOC
    "mean_abs_loc_delta",      # avg lines changed per commit
    "mean_abs_cc_delta",       # avg complexity change per commit  
    "mean_abs_imports_delta",  # avg import changes per commit
    "mean_abs_func_delta",     # avg function count change per commit
    
    # How stable are the metrics over time? (3 features)
    "std_cc_after",            # complexity bouncing around?
    "std_imports_after",       # imports bouncing around?
    "std_mi_after",            # maintainability bouncing around?
    
    # Which direction are things heading? (3 features)
    "cc_trend",                # complexity going up or down?
    "mi_trend",                # maintainability improving?
    "imports_trend",           # accumulating dependencies?
    
    # What's the absolute level? (4 features)
    "mean_cc_after",           # how complex is this file generally?
    "mean_mi_after",           # how maintainable is this file? (higher = better)
    "mean_imports_after",      # how many imports typically?
    "mean_func_after",         # how many functions?
    
    # Same questions but just for recent commits (5 features)
    "recent_cc_trend",         # complexity trend lately
    "recent_mi_trend",         # maintainability trend lately
    "recent_imports_trend",    # imports trend lately
    "recent_cc_volatility",    # is complexity stable lately?
    "recent_imports_volatility", # are imports stable lately?
]


# =============================================================================
# STEP 1: LOAD THE DATA AND GROUP BY FILE
# =============================================================================

def get_files_at_head(repo_path: Path) -> set:
    """
    Return a set of relative file paths present at the repository HEAD.
    Paths are normalized with forward slashes.
    """
    valid_exts = (".ts", ".tsx", ".js", ".jsx", ".py", ".java")
    head_files = set()

    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(valid_exts):
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                head_files.add(rel.replace("\\", "/"))

    return head_files

def load_and_prepare_data(
    min_commits: int = MIN_COMMITS_DEFAULT,
    repo_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame], Dict[str, set]]:
    """
    Load metrics and commits, merge them, group by file.
    
    The key thing here is we need timestamps to sort commits chronologically.
    Without that, "trends" don't make sense. So we join with commits.csv.
    
    We also filter out files with too few commits - can't detect patterns
    from like 2 data points.
    
    Returns a dict where each key is (repo, filepath) and value is a
    DataFrame of that file's commits sorted by time.
    """
    print(f"Loading data from {DATA_DIR}...")
    
    # Load metrics data
    metrics_df = pd.read_csv(METRICS_FILE)
    print(f"  Loaded {len(metrics_df)} metric rows")

    if repo_name:
        metrics_df = metrics_df[metrics_df["repo_name"] == repo_name]
        print(f"  Filtered to repo '{repo_name}': {len(metrics_df)} metric rows")

    # Build HEAD file index per repo
    repo_base = PROJECT_ROOT / "repos"
    repo_names = metrics_df["repo_name"].unique().tolist()
    head_files_by_repo: Dict[str, set] = {}
    for repo_name in repo_names:
        repo_path = repo_base / repo_name
        if not repo_path.exists():
            print(f"  Warning: repo path not found for {repo_name}: {repo_path}")
            head_files_by_repo[repo_name] = set()
            continue
        head_files_by_repo[repo_name] = get_files_at_head(repo_path)
    
    # Load commits data for timestamps
    commits_df = pd.read_csv(COMMITS_FILE, usecols=["repo_name", "commit_hash", "timestamp"])
    commits_df["timestamp"] = pd.to_datetime(commits_df["timestamp"], utc=True)
    print(f"  Loaded {len(commits_df)} commits")
    
    # Merge to get timestamps for each metric row
    merged_df = metrics_df.merge(
        commits_df,
        on=["repo_name", "commit_hash"],
        how="left"
    )
    
    # Some commits might not have timestamps (shouldn't happen but let's be safe)
    missing_ts = merged_df["timestamp"].isna().sum()
    if missing_ts > 0:
        print(f"  Warning: {missing_ts} rows have no matching timestamp, using commit order")
        merged_df["timestamp"] = merged_df["timestamp"].fillna(pd.Timestamp.min)
    
    # Now group everything by file and sort chronologically
    file_groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    
    for (repo, fpath), group in merged_df.groupby(["repo_name", "file_path"]):
        head_files = head_files_by_repo.get(repo, set())
        if fpath not in head_files:
            continue
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        
        # Skip files without enough history - can't see patterns in 2 commits
        if len(sorted_group) >= min_commits:
            file_groups[(repo, fpath)] = sorted_group
    
    total_files = len(merged_df.groupby(["repo_name", "file_path"]))
    included_files = len(file_groups)
    print(f"  Total unique files: {total_files}")
    print(f"  Files with >= {min_commits} commits: {included_files}")
    
    return merged_df, file_groups, head_files_by_repo


# =============================================================================
# STEP 2: TURN COMMIT HISTORY INTO FEATURES
# =============================================================================

def compute_linear_trend(series: pd.Series) -> float:
    """
    Fit a line through the values and return the slope.
    
    This tells us: is this metric going up, down, or staying flat?
    We use commit index as "time" - not perfect but works well enough.
    
    Returns 0 if there's not enough data or no variance (all same values).
    """
    clean = series.dropna()
    
    if len(clean) < 2:
        return 0.0
    
    # If every value is identical, slope is 0 (avoid numerical issues)
    if clean.std() == 0:
        return 0.0
    
    # Simple linear regression: x = commit number, y = metric value
    x = np.arange(len(clean))
    y = clean.values
    
    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope) if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def compute_file_features(
    file_groups: Dict[Tuple[str, str], pd.DataFrame],
    recent_window: int = RECENT_WINDOW_DEFAULT
) -> pd.DataFrame:
    """
    This is where the magic happens - turn a file's commit history into numbers.
    
    For each file we compute:
    - Change intensity: how much does this file get modified?
    - Volatility: are the metrics stable or bouncing around?
    - Trends: is complexity growing? maintainability dropping?
    - Structural load: how big/complex is this file in absolute terms?
    - Recent window: same questions but just for the last N commits
    
    The "recent vs historical" comparison is super useful - lets us see
    if a file that was stable is now getting messy, or vice versa.
    """
    print(f"Computing features for {len(file_groups)} files...")
    
    records = []
    
    for (repo_name, file_path), group in file_groups.items():
        n_commits = len(group)
        
        # --- How actively is this file being changed? ---
        
        # What fraction of commits actually touch this file's LOC?
        loc_changes = (group["loc_delta"] != 0).sum()
        churn_rate = loc_changes / n_commits
        
        # When it does change, how big are the changes on average?
        # (MI delta skipped - it's a composite of LOC/CC so would be redundant)
        mean_abs_loc_delta = group["loc_delta"].abs().mean()
        mean_abs_cc_delta = group["cc_delta"].abs().mean()
        mean_abs_imports_delta = group["imports_delta"].abs().mean()
        mean_abs_func_delta = group["func_delta"].abs().mean()
        
        # --- Are the metrics stable or bouncing around? ---
        
        std_cc_after = group["cc_after"].std()
        std_imports_after = group["imports_after"].std()
        std_mi_after = group["mi_after"].std()
        
        # --- Which direction are things heading over time? ---
        
        cc_trend = compute_linear_trend(group["cc_after"])
        mi_trend = compute_linear_trend(group["mi_after"])
        imports_trend = compute_linear_trend(group["imports_after"])
        
        # --- What's the typical complexity level? ---
        
        mean_cc_after = group["cc_after"].mean()
        mean_mi_after = group["mi_after"].mean()  # higher = more maintainable
        mean_imports_after = group["imports_after"].mean()
        mean_func_after = group["func_after"].mean()
        
        # --- Now the same questions but just for recent commits ---
        # This lets us catch files that WERE stable but are now degrading
        
        recent = group.tail(recent_window) if len(group) >= recent_window else group
        
        recent_cc_trend = compute_linear_trend(recent["cc_after"])
        recent_mi_trend = compute_linear_trend(recent["mi_after"])
        recent_imports_trend = compute_linear_trend(recent["imports_after"])
        
        recent_cc_volatility = recent["cc_after"].std()
        recent_imports_volatility = recent["imports_after"].std()
        # (recent_mi_volatility skipped - MI is derived from CC/LOC, would be redundant)
        
        # --- Pack it all into a record ---
        
        record = {
            "repo_name": repo_name,
            "file_path": file_path,
            "n_commits": n_commits,
            # Change Intensity
            "churn_rate": churn_rate,
            "mean_abs_loc_delta": mean_abs_loc_delta,
            "mean_abs_cc_delta": mean_abs_cc_delta,
            "mean_abs_imports_delta": mean_abs_imports_delta,
            "mean_abs_func_delta": mean_abs_func_delta,
            # Stability
            "std_cc_after": std_cc_after,
            "std_imports_after": std_imports_after,
            "std_mi_after": std_mi_after,
            # Trends
            "cc_trend": cc_trend,
            "mi_trend": mi_trend,
            "imports_trend": imports_trend,
            # Structural
            "mean_cc_after": mean_cc_after,
            "mean_mi_after": mean_mi_after,
            "mean_imports_after": mean_imports_after,
            "mean_func_after": mean_func_after,
            # Recent
            "recent_cc_trend": recent_cc_trend,
            "recent_mi_trend": recent_mi_trend,
            "recent_imports_trend": recent_imports_trend,
            "recent_cc_volatility": recent_cc_volatility,
            "recent_imports_volatility": recent_imports_volatility,
        }
        
        records.append(record)
    
    features_df = pd.DataFrame(records)
    
    # Fill any NaN values with 0 (can occur from std of single values)
    features_df[FEATURE_NAMES] = features_df[FEATURE_NAMES].fillna(0)
    
    print(f"  Generated {len(FEATURE_NAMES)} features per file")
    
    return features_df


# =============================================================================
# STEP 3: NORMALIZE SO BIG NUMBERS DON'T DOMINATE
# =============================================================================

def normalize_features(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    StandardScaler: subtract mean, divide by std for each feature.
    
    Why? Because LOC delta might be in hundreds while CC is like 3-5.
    Without normalization, LOC would dominate all the distance calculations
    and CC would basically be ignored. Not what we want.
    """
    print("Normalizing features with StandardScaler...")
    
    # Extract feature columns only
    X = features_df[FEATURE_NAMES].values
    
    # Fit and transform
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # sanity prints to verify working
    print(f"  Input shape: {X.shape}")
    print(f"  Normalized feature means near 0: {np.abs(X_normalized.mean(axis=0)).max():.6f}")
    print(f"  Normalized feature stds near 1: {np.abs(X_normalized.std(axis=0) - 1).max():.6f}")
    
    return X_normalized, scaler


# =============================================================================
# STEP 4a: GROUP SIMILAR FILES TOGETHER
# =============================================================================

def run_clustering(X_normalized: np.ndarray) -> Tuple[np.ndarray, int, KMeans]:
    """
    KMeans clustering to find files that evolve similarly.
    
    We don't know how many clusters there should be, so we try k=3 to k=10
    and pick whichever gives the best silhouette score (higher = better
    separated clusters).
    
    The idea is: maybe there's a group of "stable core files", a group of
    "frequently refactored files", etc. Clustering finds these groups.
    """
    print(f"Running KMeans clustering (k={K_MIN}-{K_MAX})...")
    
    n_samples = X_normalized.shape[0]
    
    # Adjust k range if we have too few samples
    k_min = min(K_MIN, n_samples - 1)
    k_max = min(K_MAX, n_samples - 1)
    
    if k_min < 2:
        print("  Warning: Too few samples for meaningful clustering")
        # Return all in one cluster
        return np.zeros(n_samples, dtype=int), 1, None
    
    best_k = k_min
    best_score = -1
    best_model = None
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_normalized)
        
        # Compute silhouette score
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_normalized, labels)
        else:
            score = -1
        
        print(f"    k={k}: silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
    
    print(f"  Selected k={best_k} (silhouette={best_score:.4f})")
    
    return best_model.labels_, best_k, best_model


# =============================================================================
# STEP 4b: FIND THE WEIRD FILES
# =============================================================================

def run_anomaly_detection(X_normalized: np.ndarray) -> np.ndarray:
    """
    Isolation Forest to find files with unusual evolution patterns.
    
    The algorithm basically asks: how easy is it to isolate this point?
    Normal files cluster together and take many splits to isolate.
    Weird files are easy to separate = high anomaly score.
    
    Output is normalized to 0-1 where 1 = "this file is WEIRD".
    
    Examples of what might score high:
    - File that oscillates between two completely different states
    - File with extremely high churn compared to peers
    - File with unusual combination of stable LOC but volatile complexity
    """
    print("Running Isolation Forest anomaly detection...")
    
    iso_forest = IsolationForest(
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100
    )
    
    # Fit and get raw scores
    iso_forest.fit(X_normalized)
    raw_scores = iso_forest.decision_function(X_normalized)
    
    # decision_function returns negative scores for anomalies
    # Convert to [0, 1] where 1 = most anomalous
    # Lower raw_score = more anomalous, so we negate and normalize
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    if max_score == min_score:
        # All same score (unlikely)
        normalized_scores = np.zeros_like(raw_scores)
    else:
        # Negate so higher = more anomalous, then normalize to [0, 1]
        normalized_scores = (max_score - raw_scores) / (max_score - min_score)
    
    n_anomalies = (iso_forest.predict(X_normalized) == -1).sum()
    print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(raw_scores)*100:.1f}%)")
    
    return normalized_scores


# =============================================================================
# STEP 5: TURN NUMBERS INTO HUMAN-READABLE LABELS
# =============================================================================

# These are simple threshold-based rules - no ML, just if/else.
# The thresholds are defined at the top of the file so they're easy to tune.

def classify_churn(churn_rate: float) -> str:
    """low/moderate/high based on what fraction of commits change the file"""
    if churn_rate <= CHURN_THRESHOLDS[0]:
        return "low"
    elif churn_rate <= CHURN_THRESHOLDS[1]:
        return "moderate"
    else:
        return "high"


def classify_trend(trend_value: float, threshold: float) -> str:
    """Is the metric going up, down, or staying flat?"""
    if trend_value < -threshold:
        return "decreasing"
    elif trend_value > threshold:
        return "increasing"
    else:
        return "stable"


def classify_volatility(std_value: float) -> str:
    """Is the metric stable or bouncing around?"""
    if std_value <= VOLATILITY_THRESHOLD:
        return "stable"
    else:
        return "oscillating"


def compare_recent_to_historical(recent: float, historical: float) -> str:
    """
    Is the file getting better or worse lately compared to its history?
    
    "improving" = recent trend is better (lower complexity, higher MI, etc)
    "regressing" = recent trend is worse
    "unchanged" = about the same
    """
    if historical == 0:
        if recent == 0:
            return "unchanged"
        return "regressing" if recent > 0 else "improving"
    
    relative_change = (recent - historical) / abs(historical)
    
    if abs(relative_change) <= RECENT_COMPARISON_THRESHOLD:
        return "unchanged"
    elif relative_change > 0:
        return "regressing"
    else:
        return "improving"


def apply_rule_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take all the numeric features and add human-readable labels.
    
    After this, each file has labels like:
    - churn_label: "high" 
    - cc_trend_label: "increasing"
    - recent_cc_trend_vs_historical: "regressing"
    
    These are what the LLM will use to generate explanations.
    """
    print("Applying rule-based labels...")
    
    df = features_df.copy()
    
    # Churn label
    df["churn_label"] = df["churn_rate"].apply(classify_churn)
    
    # Trend labels (historical)
    df["cc_trend_label"] = df["cc_trend"].apply(
        lambda x: classify_trend(x, CC_TREND_THRESHOLD)
    )
    df["mi_trend_label"] = df["mi_trend"].apply(
        lambda x: classify_trend(x, MI_TREND_THRESHOLD)
    )
    df["imports_trend_label"] = df["imports_trend"].apply(
        lambda x: classify_trend(x, IMPORTS_TREND_THRESHOLD)
    )
    
    # Volatility labels
    df["cc_volatility_label"] = df["std_cc_after"].apply(classify_volatility)
    df["imports_volatility_label"] = df["std_imports_after"].apply(classify_volatility)
    
    # Recent vs historical comparisons
    df["recent_cc_trend_vs_historical"] = df.apply(
        lambda row: compare_recent_to_historical(
            row["recent_cc_trend"], row["cc_trend"]
        ),
        axis=1
    )
    df["recent_imports_trend_vs_historical"] = df.apply(
        lambda row: compare_recent_to_historical(
            row["recent_imports_trend"], row["imports_trend"]
        ),
        axis=1
    )
    df["recent_cc_volatility_vs_historical"] = df.apply(
        lambda row: compare_recent_to_historical(
            row["recent_cc_volatility"], row["std_cc_after"]
        ),
        axis=1
    )
    
    print("  Added 9 rule-based label columns")
    
    return df


def derive_cluster_label(cluster_stats: Dict[str, float]) -> str:
    """
    Give each cluster a human-readable name based on its characteristics.
    
    Examples:
    - "high churn, stable complexity" = actively developed but well-maintained
    - "low churn, growing complexity" = neglected and rotting
    - "moderate churn, declining complexity, high volatility" = being refactored
    """
    parts = []
    
    # Churn characterization
    churn = cluster_stats.get("avg_churn_rate", 0)
    if churn > CHURN_THRESHOLDS[1]:
        parts.append("high churn")
    elif churn > CHURN_THRESHOLDS[0]:
        parts.append("moderate churn")
    else:
        parts.append("low churn")
    
    # Complexity trend
    cc_trend = cluster_stats.get("avg_cc_trend", 0)
    if cc_trend > CC_TREND_THRESHOLD:
        parts.append("growing complexity")
    elif cc_trend < -CC_TREND_THRESHOLD:
        parts.append("declining complexity")
    else:
        parts.append("stable complexity")
    
    # Volatility
    cc_vol = cluster_stats.get("avg_std_cc_after", 0)
    if cc_vol > VOLATILITY_THRESHOLD:
        parts.append("high volatility")
    
    return ", ".join(parts)


def summarize_clusters(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    n_clusters: int
) -> List[Dict[str, Any]]:
    """
    For each cluster, compute aggregate stats and give it a label.
    
    This helps understand what each cluster represents:
    - How many files are in it?
    - What's the average churn/complexity/etc?
    - What's the distribution of labels within the cluster?
    """
    print(f"Summarizing {n_clusters} clusters...")
    
    df = features_df.copy()
    df["cluster_id"] = cluster_labels
    
    summaries = []
    
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cluster_id]
        
        stats = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_df),
            "avg_churn_rate": float(cluster_df["churn_rate"].mean()),
            "avg_cc_after": float(cluster_df["mean_cc_after"].mean()),
            "avg_cc_trend": float(cluster_df["cc_trend"].mean()),
            "avg_mi_after": float(cluster_df["mean_mi_after"].mean()),
            "avg_mi_trend": float(cluster_df["mi_trend"].mean()),
            "avg_mi_volatility": float(cluster_df["std_mi_after"].mean()),
            "avg_imports_after": float(cluster_df["mean_imports_after"].mean()),
            "avg_imports_trend": float(cluster_df["imports_trend"].mean()),
            "avg_std_cc_after": float(cluster_df["std_cc_after"].mean()),
            "avg_std_imports_after": float(cluster_df["std_imports_after"].mean()),
            "avg_func_after": float(cluster_df["mean_func_after"].mean()),
            # What labels are in this cluster?
            "churn_label_distribution": cluster_df["churn_label"].value_counts().to_dict(),
            "cc_trend_label_distribution": cluster_df["cc_trend_label"].value_counts().to_dict(),
        }
        
        # Derive cluster label
        stats["derived_cluster_label"] = derive_cluster_label(stats)
        
        summaries.append(stats)
        print(f"  Cluster {cluster_id}: {stats['size']} files - {stats['derived_cluster_label']}")
    
    return summaries


# =============================================================================
# STEP 6: PACKAGE EVERYTHING INTO JSON
# =============================================================================

def build_file_output(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Build the final per-file output that gets written to JSON.
    
    Each file gets a nice structured object with:
    - Identifiers (repo, path, commit count)
    - ML results (cluster, anomaly score)
    - Labels (churn, trends, volatility)
    - Raw numbers (for verification/debugging)
    
    This is what the LLM explanation layer will consume.
    """
    outputs = []
    
    for idx, row in features_df.iterrows():
        output = {
            "repo_name": row["repo_name"],
            "file_path": row["file_path"],
            "n_commits": int(row["n_commits"]),
            "cluster_id": int(cluster_labels[idx]),
            "anomaly_score": float(round(anomaly_scores[idx], 4)),
            
            # Labels
            "churn_label": row["churn_label"],
            "cc_volatility_label": row["cc_volatility_label"],
            "imports_volatility_label": row["imports_volatility_label"],
            
            # Historical trends
            "historical_trends": {
                "cc": {
                    "value": float(round(row["cc_trend"], 6)),
                    "label": row["cc_trend_label"]
                },
                "mi": {
                    "value": float(round(row["mi_trend"], 6)),
                    "label": row["mi_trend_label"]
                },
                "imports": {
                    "value": float(round(row["imports_trend"], 6)),
                    "label": row["imports_trend_label"]
                }
            },
            
            # Recent trends
            "recent_trends": {
                "cc": {
                    "value": float(round(row["recent_cc_trend"], 6)),
                    "vs_historical": row["recent_cc_trend_vs_historical"]
                },
                "mi": {
                    "value": float(round(row["recent_mi_trend"], 6)),
                },
                "imports": {
                    "value": float(round(row["recent_imports_trend"], 6)),
                    "vs_historical": row["recent_imports_trend_vs_historical"]
                }
            },
            
            # Volatility comparison
            "volatility_comparison": {
                "cc": row["recent_cc_volatility_vs_historical"]
            },
            
            # Raw feature values for reference
            "raw_features": {
                "churn_rate": float(round(row["churn_rate"], 4)),
                "mean_abs_loc_delta": float(round(row["mean_abs_loc_delta"], 4)),
                "mean_cc_after": float(round(row["mean_cc_after"], 4)),
                "mean_mi_after": float(round(row["mean_mi_after"], 4)),
                "mean_imports_after": float(round(row["mean_imports_after"], 4)),
                "mean_func_after": float(round(row["mean_func_after"], 4)),
                "std_cc_after": float(round(row["std_cc_after"], 4)),
                "std_imports_after": float(round(row["std_imports_after"], 4)),
            }
        }
        
        outputs.append(output)
    
    return outputs


def export_results(
    file_outputs: List[Dict[str, Any]],
    cluster_summaries: List[Dict[str, Any]],
    output_file_analysis: Path = OUTPUT_FILE_ANALYSIS,
    output_cluster_summary: Path = OUTPUT_CLUSTER_SUMMARY
) -> None:
    """
    Write everything to JSON files in the data/ folder.
    
    Two files:
    - file_analysis.json: one entry per file with all the details
    - cluster_summary.json: one entry per cluster with aggregate stats
    """
    print(f"Exporting results...")
    
    # Export file analysis
    with open(output_file_analysis, "w", encoding="utf-8") as f:
        json.dump(file_outputs, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(file_outputs)} file records to {output_file_analysis}")
    
    # Export cluster summaries
    with open(output_cluster_summary, "w", encoding="utf-8") as f:
        json.dump(cluster_summaries, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(cluster_summaries)} cluster records to {output_cluster_summary}")


# =============================================================================
# RUN THE WHOLE THING
# =============================================================================

def main(
    min_commits: int = MIN_COMMITS_DEFAULT,
    recent_window: int = RECENT_WINDOW_DEFAULT,
    repo_name: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the full pipeline from raw data to JSON output.
    
    Steps:
    1. Load data, group by file
    2. Compute features for each file
    3. Normalize (so clustering works properly)
    4. Cluster + anomaly detection
    5. Apply rule-based labels
    6. Export to JSON
    
    Returns the outputs so you can also use this programmatically.
    """
    print("=" * 60)
    print("ML Pipeline for Temporal Code Evolution Analysis")
    print("=" * 60)
    print(f"Config: min_commits={min_commits}, recent_window={recent_window}, repo={repo_name or 'ALL'}")
    print()
    
    # Load everything and group by file
    merged_df, file_groups, head_files_by_repo = load_and_prepare_data(
        min_commits=min_commits,
        repo_name=repo_name
    )
    
    if len(file_groups) == 0:
        print("\nNo files have enough commits to analyze. Try lowering --min-commits?")
        return [], []
    
    print()
    
    # Turn commit history into feature vectors
    features_df = compute_file_features(file_groups, recent_window=recent_window)
    print()
    
    # Normalize so clustering doesn't get dominated by big numbers
    X_normalized, scaler = normalize_features(features_df)
    print()
    
    # Find groups of similar files
    cluster_labels, n_clusters, kmeans_model = run_clustering(X_normalized)
    print()
    
    # Flag the weird ones
    anomaly_scores = run_anomaly_detection(X_normalized)
    print()
    
    # Turn numbers into words
    labeled_df = apply_rule_labels(features_df)
    print()
    
    # Summarize what each cluster looks like
    cluster_summaries = summarize_clusters(labeled_df, cluster_labels, n_clusters)
    print()
    
    # Package it all up and write to JSON
    file_outputs = build_file_output(labeled_df, cluster_labels, anomaly_scores)

    # Defensive assertion: ensure only HEAD files reach ML outputs
    for f in file_outputs:
        repo_name = f["repo_name"]
        file_path = f["file_path"]
        assert file_path in head_files_by_repo.get(repo_name, set()), (
            f"Deleted file leaked into ML outputs: {file_path}"
        )
    export_results(file_outputs, cluster_summaries)
    
    print()
    print("=" * 60)
    print("Done! Check data/file_analysis.json and data/cluster_summary.json")
    print("=" * 60)
    
    return file_outputs, cluster_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Temporal Code Evolution Analysis"
    )
    parser.add_argument(
        "--min-commits",
        type=int,
        default=MIN_COMMITS_DEFAULT,
        help=f"Minimum commits per file (default: {MIN_COMMITS_DEFAULT})"
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=RECENT_WINDOW_DEFAULT,
        help=f"Recent window size for trend features (default: {RECENT_WINDOW_DEFAULT})"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Optional repo name to analyze (matches repo_name in metrics)"
    )
    
    args = parser.parse_args()
    
    main(
        min_commits=args.min_commits,
        recent_window=args.recent_window,
        repo_name=args.repo
    )