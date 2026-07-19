import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "nextjs_outputs"

FILE_ANALYSIS_PATH = DATA_DIR / "file_analysis.json"
CLUSTER_SUMMARY_PATH = DATA_DIR / "cluster_summary.json"
REPOSITORY_REPORT_PATH = OUTPUT_DIR / "repository_report.json"


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

@st.cache_data
def load_file_analysis() -> pd.DataFrame:
    with open(FILE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        file_analysis = json.load(f)

    df = pd.DataFrame(file_analysis)

    # --- Flatten raw_features ---
    raw_df = pd.json_normalize(df["raw_features"])
    raw_df.columns = [f"raw_{c}" for c in raw_df.columns]

    # --- Flatten historical_trends ---
    hist_df = pd.json_normalize(df["historical_trends"])
    hist_df.columns = [f"hist_{c.replace('.', '_')}" for c in hist_df.columns]

    # --- Flatten recent_trends ---
    recent_df = pd.json_normalize(df["recent_trends"])
    recent_df.columns = [f"recent_{c.replace('.', '_')}" for c in recent_df.columns]

    df = pd.concat(
        [
            df.drop(
                columns=[
                    "raw_features",
                    "historical_trends",
                    "recent_trends",
                ]
            ),
            raw_df,
            hist_df,
            recent_df,
        ],
        axis=1,
    )

    return df


@st.cache_data
def load_cluster_summary() -> pd.DataFrame:
    with open(CLUSTER_SUMMARY_PATH, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    return pd.DataFrame(clusters)


@st.cache_data
def load_repository_report() -> dict:
    with open(REPOSITORY_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Code Forensics Dashboard",
    layout="wide",
)

st.title("Code Forensics - Evolution Dashboard")
st.caption(
    "Temporal code evolution analysis. Descriptive, not prescriptive."
)

df = load_file_analysis()
clusters_df = load_cluster_summary()
report = load_repository_report()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------

st.sidebar.header("Filters")

repos = sorted(df["repo_name"].unique())
selected_repo = st.sidebar.selectbox("Repository", repos)

filtered_df = df[df["repo_name"] == selected_repo]

clusters = sorted(filtered_df["cluster_id"].unique())
selected_clusters = st.sidebar.multiselect(
    "Clusters",
    clusters,
    default=clusters,
)

filtered_df = filtered_df[
    filtered_df["cluster_id"].isin(selected_clusters)
]

min_anomaly = st.sidebar.slider(
    "Minimum anomaly score",
    0.0,
    1.0,
    0.0,
    0.05,
)

filtered_df = filtered_df[
    filtered_df["anomaly_score"] >= min_anomaly
]

# -----------------------------------------------------------------------------
# OVERVIEW METRICS
# -----------------------------------------------------------------------------

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Files Analyzed",
    len(filtered_df),
)

col2.metric(
    "Avg Anomaly Score",
    round(filtered_df["anomaly_score"].mean(), 3)
    if len(filtered_df) > 0
    else 0.0,
)

col3.metric(
    "Clusters Present",
    filtered_df["cluster_id"].nunique(),
)

# -----------------------------------------------------------------------------
# REPOSITORY SYNTHESIS REPORT
# -----------------------------------------------------------------------------

st.subheader("Repository Analysis Report")

if report:
    st.markdown(report.get("synthesis", "No synthesis available"))
    
    with st.expander("Analysis Metadata"):
        metadata = report.get("metadata", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files Analyzed", metadata.get("files_analyzed", 0))
        col2.metric("Clusters", metadata.get("clusters", 0))
        col3.metric("Model", metadata.get("model", "N/A"))
        col4.metric("Temperature", metadata.get("temperature", 0))
        
        st.subheader("Grounding Metrics")
        grounding = metadata.get("grounding_metrics", {})
        gr_col1, gr_col2, gr_col3 = st.columns(3)
        gr_col1.metric("Commit Grounding Rate", f"{grounding.get('commit_grounding_rate', 0):.1f}%")
        gr_col2.metric("Quantified Comparison Rate", f"{grounding.get('quantified_comparison_rate', 0):.1f}%")
        gr_col3.metric("Word Limit Compliance", f"{grounding.get('word_limit_compliance', 0):.1f}%")

# -----------------------------------------------------------------------------
# SCATTER: CHURN vs ANOMALY
# -----------------------------------------------------------------------------

st.subheader("Churn vs Anomaly")

scatter_fig = px.scatter(
    filtered_df,
    x="raw_churn_rate",
    y="anomaly_score",
    color="cluster_id",
    hover_data=["file_path", "n_commits"],
    labels={
        "raw_churn_rate": "Churn Rate",
        "anomaly_score": "Anomaly Score",
        "cluster_id": "Cluster",
    },
)

st.plotly_chart(scatter_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SCATTER: COMPLEXITY TREND vs CHURN
# -----------------------------------------------------------------------------

st.subheader("Complexity Trend vs Churn")

complexity_fig = px.scatter(
    filtered_df,
    x="raw_churn_rate",
    y="hist_cc_value",
    color="cluster_id",
    hover_data=["file_path"],
    labels={
        "raw_churn_rate": "Churn Rate",
        "hist_cc_value": "Complexity Trend (Slope)",
    },
)

st.plotly_chart(complexity_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# CLUSTER DISTRIBUTION
# -----------------------------------------------------------------------------

st.subheader("Cluster Distribution")

cluster_counts = (
    filtered_df
    .groupby("cluster_id", as_index=False)
    .size()
    .rename(columns={"size": "file_count"})
)


cluster_bar = px.bar(
    cluster_counts,
    x="cluster_id",
    y="file_count",
    labels={
        "cluster_id": "Cluster",
        "file_count": "Files",
    },
)


st.plotly_chart(cluster_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# FILE TABLE
# -----------------------------------------------------------------------------

st.subheader("File-Level Details")

display_cols = [
    "file_path",
    "cluster_id",
    "anomaly_score",
    "raw_churn_rate",
    "hist_cc_value",
    "n_commits",
]

st.dataframe(
    filtered_df[display_cols]
    .sort_values("anomaly_score", ascending=False),
    use_container_width=True,
)
