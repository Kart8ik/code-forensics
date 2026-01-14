"""
ML Pipeline for Temporal Code Evolution Analysis

This module implements an unsupervised ML pipeline that:
1. Aggregates commit-level data into per-file temporal summaries
2. Applies clustering and anomaly detection at the file level
3. Produces structured, machine-readable JSON outputs

The pipeline operates per-file (NOT per-repo) and focuses on:
- Evolution patterns
- Stability analysis  
- Deviation detection

Output feeds a downstream LLM explanation layer.
"""

import argparse
import json
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
# CONFIGURATION CONSTANTS
# =============================================================================

# Minimum commits required per file to be included in analysis
MIN_COMMITS_DEFAULT = 5

# Recent window size for computing recent trends/volatility
RECENT_WINDOW_DEFAULT = 20

# KMeans cluster range for auto-selection via silhouette score
K_MIN = 3
K_MAX = 10

# Isolation Forest contamination (expected proportion of outliers)
ISOLATION_CONTAMINATION = 0.1

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# THRESHOLD CONSTANTS FOR RULE-BASED LABELS
# =============================================================================

# Churn rate thresholds: (low_upper, moderate_upper)
# low: <= 0.3, moderate: 0.3-0.7, high: > 0.7
CHURN_THRESHOLDS = (0.3, 0.7)

# CC trend thresholds for directional labeling
# decreasing: < -threshold, stable: within threshold, increasing: > threshold
CC_TREND_THRESHOLD = 0.01

# MI trend thresholds
MI_TREND_THRESHOLD = 0.1

# Imports trend thresholds
IMPORTS_TREND_THRESHOLD = 0.05

# Volatility thresholds (std deviation)
# stable: <= threshold, oscillating: > threshold
VOLATILITY_THRESHOLD = 1.0

# Recent vs historical comparison threshold (relative change)
# improving/regressing if |recent - historical| > threshold * historical
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
# FEATURE NAMES (for documentation and indexing)
# =============================================================================

FEATURE_NAMES = [
    # Change Intensity (5)
    "churn_rate",
    "mean_abs_loc_delta",
    "mean_abs_cc_delta",
    "mean_abs_imports_delta",
    "mean_abs_func_delta",
    # Stability/Volatility (3)
    "std_cc_after",
    "std_imports_after",
    "std_mi_after",
    # Directional Trends (3)
    "cc_trend",
    "mi_trend",
    "imports_trend",
    # Structural Load (3)
    "mean_cc_after",
    "mean_imports_after",
    "mean_func_after",
    # Recent Window Features (5)
    "recent_cc_trend",
    "recent_mi_trend",
    "recent_imports_trend",
    "recent_cc_volatility",
    "recent_imports_volatility",
]


# =============================================================================
# STEP 1: DATA PREPARATION
# =============================================================================

def load_and_prepare_data(
    min_commits: int = MIN_COMMITS_DEFAULT
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    Load and prepare data for ML pipeline.
    
    Steps:
    1. Load metrics_all.csv and commits.csv
    2. Join on (repo_name, commit_hash) to get timestamps
    3. Group by (repo_name, file_path)
    4. Sort each group by timestamp
    5. Filter files with fewer than min_commits changes
    
    Args:
        min_commits: Minimum number of commits required per file
        
    Returns:
        Tuple of:
        - Full merged DataFrame
        - Dictionary mapping (repo_name, file_path) -> sorted DataFrame
    """
    print(f"Loading data from {DATA_DIR}...")
    
    # Load metrics data
    metrics_df = pd.read_csv(METRICS_FILE)
    print(f"  Loaded {len(metrics_df)} metric rows")
    
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
    
    # Handle any rows without matching timestamps
    missing_ts = merged_df["timestamp"].isna().sum()
    if missing_ts > 0:
        print(f"  Warning: {missing_ts} rows have no matching timestamp, using commit order")
        # For rows without timestamps, we'll rely on the original order
        merged_df["timestamp"] = merged_df["timestamp"].fillna(pd.Timestamp.min)
    
    # Group by (repo_name, file_path) and sort by timestamp
    file_groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    
    for (repo, fpath), group in merged_df.groupby(["repo_name", "file_path"]):
        # Sort by timestamp (commit order)
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        
        # Filter: only include files with enough commits
        if len(sorted_group) >= min_commits:
            file_groups[(repo, fpath)] = sorted_group
    
    total_files = len(merged_df.groupby(["repo_name", "file_path"]))
    included_files = len(file_groups)
    print(f"  Total unique files: {total_files}")
    print(f"  Files with >= {min_commits} commits: {included_files}")
    
    return merged_df, file_groups


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def compute_linear_trend(series: pd.Series) -> float:
    """
    Compute linear regression slope over a numeric series.
    
    Uses commit index as the independent variable (proxy for time).
    Returns 0.0 if series has insufficient variance or length.
    
    Args:
        series: Pandas Series of numeric values
        
    Returns:
        Slope of linear regression (rate of change per commit)
    """
    # Remove NaN values
    clean = series.dropna()
    
    if len(clean) < 2:
        return 0.0
    
    # Check for zero variance (all identical values)
    if clean.std() == 0:
        return 0.0
    
    # Use commit index as x-axis
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
    Compute feature vectors for each file based on its temporal history.
    
    Feature categories:
    A. Change Intensity - How actively the file is modified
    B. Stability/Volatility - How much metrics fluctuate
    C. Directional Trends - Long-term trajectory of metrics
    D. Structural Load - Absolute complexity levels
    E. Recent Window - Same as B/C but on recent commits only
    
    Args:
        file_groups: Dictionary from load_and_prepare_data()
        recent_window: Number of recent commits for window features
        
    Returns:
        DataFrame with one row per file, columns are features
    """
    print(f"Computing features for {len(file_groups)} files...")
    
    records = []
    
    for (repo_name, file_path), group in file_groups.items():
        n_commits = len(group)
        
        # ------------------------------------------------------------------
        # A. CHANGE INTENSITY FEATURES
        # ------------------------------------------------------------------
        
        # churn_rate: proportion of commits with non-zero LOC change
        loc_changes = (group["loc_delta"] != 0).sum()
        churn_rate = loc_changes / n_commits
        
        # Mean absolute deltas (magnitude of changes)
        mean_abs_loc_delta = group["loc_delta"].abs().mean()
        mean_abs_cc_delta = group["cc_delta"].abs().mean()
        mean_abs_imports_delta = group["imports_delta"].abs().mean()
        mean_abs_func_delta = group["func_delta"].abs().mean()
        
        # ------------------------------------------------------------------
        # B. STABILITY / VOLATILITY FEATURES
        # ------------------------------------------------------------------
        
        std_cc_after = group["cc_after"].std()
        std_imports_after = group["imports_after"].std()
        std_mi_after = group["mi_after"].std()
        
        # ------------------------------------------------------------------
        # C. DIRECTIONAL TRENDS (linear regression slopes)
        # ------------------------------------------------------------------
        
        cc_trend = compute_linear_trend(group["cc_after"])
        mi_trend = compute_linear_trend(group["mi_after"])
        imports_trend = compute_linear_trend(group["imports_after"])
        
        # ------------------------------------------------------------------
        # D. STRUCTURAL LOAD (absolute levels)
        # ------------------------------------------------------------------
        
        mean_cc_after = group["cc_after"].mean()
        mean_imports_after = group["imports_after"].mean()
        mean_func_after = group["func_after"].mean()
        
        # ------------------------------------------------------------------
        # E. RECENT WINDOW FEATURES
        # ------------------------------------------------------------------
        
        # Get recent commits (last N)
        recent = group.tail(recent_window) if len(group) >= recent_window else group
        
        # Recent trends (using same linear regression approach)
        recent_cc_trend = compute_linear_trend(recent["cc_after"])
        recent_mi_trend = compute_linear_trend(recent["mi_after"])
        recent_imports_trend = compute_linear_trend(recent["imports_after"])
        
        # Recent volatility
        recent_cc_volatility = recent["cc_after"].std()
        recent_imports_volatility = recent["imports_after"].std()
        
        # ------------------------------------------------------------------
        # BUILD FEATURE RECORD
        # ------------------------------------------------------------------
        
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
# STEP 3: NORMALIZATION
# =============================================================================

def normalize_features(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize feature vectors using StandardScaler.
    
    This ensures features with different magnitudes do not dominate
    the distance calculations in clustering and anomaly detection.
    
    Args:
        features_df: DataFrame from compute_file_features()
        
    Returns:
        Tuple of:
        - Normalized feature matrix (n_files x n_features)
        - Fitted StandardScaler (for inverse transform if needed)
    """
    print("Normalizing features with StandardScaler...")
    
    # Extract feature columns only
    X = features_df[FEATURE_NAMES].values
    
    # Fit and transform
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    print(f"  Input shape: {X.shape}")
    print(f"  Normalized feature means near 0: {np.abs(X_normalized.mean(axis=0)).max():.6f}")
    print(f"  Normalized feature stds near 1: {np.abs(X_normalized.std(axis=0) - 1).max():.6f}")
    
    return X_normalized, scaler


# =============================================================================
# STEP 4a: CLUSTERING
# =============================================================================

def run_clustering(X_normalized: np.ndarray) -> Tuple[np.ndarray, int, KMeans]:
    """
    Cluster files using KMeans with automatic k selection.
    
    Selects optimal k by maximizing silhouette score over range [K_MIN, K_MAX].
    
    Args:
        X_normalized: Normalized feature matrix
        
    Returns:
        Tuple of:
        - Cluster labels (n_files,)
        - Optimal k value
        - Fitted KMeans model
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
# STEP 4b: ANOMALY DETECTION
# =============================================================================

def run_anomaly_detection(X_normalized: np.ndarray) -> np.ndarray:
    """
    Detect anomalous files using Isolation Forest.
    
    Returns anomaly scores where higher (less negative) = more anomalous.
    Scores are normalized to [0, 1] range for interpretability.
    
    Args:
        X_normalized: Normalized feature matrix
        
    Returns:
        Anomaly scores (n_files,) in range [0, 1]
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
# STEP 5: RULE-BASED INTERPRETATION
# =============================================================================

def classify_churn(churn_rate: float) -> str:
    """Classify churn rate into categorical label."""
    if churn_rate <= CHURN_THRESHOLDS[0]:
        return "low"
    elif churn_rate <= CHURN_THRESHOLDS[1]:
        return "moderate"
    else:
        return "high"


def classify_trend(trend_value: float, threshold: float) -> str:
    """Classify trend into directional label."""
    if trend_value < -threshold:
        return "decreasing"
    elif trend_value > threshold:
        return "increasing"
    else:
        return "stable"


def classify_volatility(std_value: float) -> str:
    """Classify volatility into categorical label."""
    if std_value <= VOLATILITY_THRESHOLD:
        return "stable"
    else:
        return "oscillating"


def compare_recent_to_historical(recent: float, historical: float) -> str:
    """
    Compare recent metric to historical.
    
    For trends: positive trend = complexity increasing = regressing
    For volatility: higher = regressing
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
    Apply rule-based labels to feature values.
    
    Adds categorical columns based on transparent thresholds.
    
    Args:
        features_df: DataFrame with computed features
        
    Returns:
        DataFrame with additional label columns
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
    Derive a descriptive label for a cluster based on its aggregate stats.
    
    Uses a decision tree approach based on dominant characteristics.
    
    Args:
        cluster_stats: Dictionary of average feature values for cluster
        
    Returns:
        Short descriptive label (e.g., "high churn, bounded complexity")
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
    Generate summary statistics for each cluster.
    
    Args:
        features_df: DataFrame with features and rule labels
        cluster_labels: Cluster assignments
        n_clusters: Number of clusters
        
    Returns:
        List of cluster summary dictionaries
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
            "avg_mi_after": float(cluster_df["mean_cc_after"].mean()),
            "avg_mi_trend": float(cluster_df["mi_trend"].mean()),
            "avg_imports_after": float(cluster_df["mean_imports_after"].mean()),
            "avg_imports_trend": float(cluster_df["imports_trend"].mean()),
            "avg_std_cc_after": float(cluster_df["std_cc_after"].mean()),
            "avg_std_imports_after": float(cluster_df["std_imports_after"].mean()),
            "avg_func_after": float(cluster_df["mean_func_after"].mean()),
            # Distribution of labels
            "churn_label_distribution": cluster_df["churn_label"].value_counts().to_dict(),
            "cc_trend_label_distribution": cluster_df["cc_trend_label"].value_counts().to_dict(),
        }
        
        # Derive cluster label
        stats["derived_cluster_label"] = derive_cluster_label(stats)
        
        summaries.append(stats)
        print(f"  Cluster {cluster_id}: {stats['size']} files - {stats['derived_cluster_label']}")
    
    return summaries


# =============================================================================
# STEP 6: OUTPUT GENERATION
# =============================================================================

def build_file_output(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Build per-file output dictionaries.
    
    Args:
        features_df: DataFrame with features and labels
        cluster_labels: Cluster assignments
        anomaly_scores: Anomaly scores
        
    Returns:
        List of file analysis dictionaries
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
    Export results to JSON files.
    
    Args:
        file_outputs: Per-file analysis results
        cluster_summaries: Per-cluster summaries
        output_file_analysis: Path for file analysis JSON
        output_cluster_summary: Path for cluster summary JSON
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
# MAIN ORCHESTRATION
# =============================================================================

def main(
    min_commits: int = MIN_COMMITS_DEFAULT,
    recent_window: int = RECENT_WINDOW_DEFAULT
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the complete ML pipeline.
    
    Args:
        min_commits: Minimum commits per file to include
        recent_window: Number of recent commits for window features
        
    Returns:
        Tuple of (file_outputs, cluster_summaries)
    """
    print("=" * 60)
    print("ML Pipeline for Temporal Code Evolution Analysis")
    print("=" * 60)
    print(f"Parameters: min_commits={min_commits}, recent_window={recent_window}")
    print()
    
    # Step 1: Load and prepare data
    merged_df, file_groups = load_and_prepare_data(min_commits=min_commits)
    
    if len(file_groups) == 0:
        print("\nNo files with sufficient commit history. Exiting.")
        return [], []
    
    print()
    
    # Step 2: Feature engineering
    features_df = compute_file_features(file_groups, recent_window=recent_window)
    print()
    
    # Step 3: Normalization
    X_normalized, scaler = normalize_features(features_df)
    print()
    
    # Step 4a: Clustering
    cluster_labels, n_clusters, kmeans_model = run_clustering(X_normalized)
    print()
    
    # Step 4b: Anomaly detection
    anomaly_scores = run_anomaly_detection(X_normalized)
    print()
    
    # Step 5: Rule-based labels
    labeled_df = apply_rule_labels(features_df)
    print()
    
    # Step 5b: Cluster summaries
    cluster_summaries = summarize_clusters(labeled_df, cluster_labels, n_clusters)
    print()
    
    # Step 6: Build and export outputs
    file_outputs = build_file_output(labeled_df, cluster_labels, anomaly_scores)
    export_results(file_outputs, cluster_summaries)
    
    print()
    print("=" * 60)
    print("Pipeline complete!")
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
    
    args = parser.parse_args()
    
    main(min_commits=args.min_commits, recent_window=args.recent_window)
