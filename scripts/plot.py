import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "hotspots.csv")



# ---------------------------
# Plot 1: Churn vs Complexity
# ---------------------------
plt.figure()
plt.scatter(df["change_count"], df["cc_after"])
plt.xlabel("Change Count (Churn)")
plt.ylabel("Cyclomatic Complexity (After)")
plt.title("Churn vs Complexity (All Repositories)")
plt.tight_layout()
plt.savefig(FIG_DIR / "churn_vs_complexity.png")
plt.close()

# ---------------------------
# Plot 2: Top 10 Hotspots
# ---------------------------
top10 = df.sort_values("risk_score", ascending=False).head(10)

plt.figure()
plt.barh(top10["file_path"], top10["risk_score"])
plt.xlabel("Risk Score")
plt.ylabel("File Path")
plt.title("Top 10 Code Hotspots")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "top10_hotspots.png")
plt.close()

# ---------------------------
# Plot 3: Repo-wise Risk
# ---------------------------
plt.figure()
df.boxplot(column="risk_score", by="repo_name")
plt.title("Risk Score Distribution by Repository")
plt.suptitle("")
plt.xlabel("Repository")
plt.ylabel("Risk Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "repo_risk_distribution.png")
plt.close()


features = df[
    ["change_count", "cc_delta", "loc_delta", "risk_score"]
].fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# -----------------------
# KMeans (Risk Clustering)
# -----------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["risk_cluster"] = kmeans.fit_predict(X)

cluster_output = DATA_DIR / "hotspots_with_clusters.csv"
df.to_csv(cluster_output, index=False)

print("KMeans clustering complete")
print(df["risk_cluster"].value_counts())

# -----------------------
# Isolation Forest (Anomalies)
# -----------------------
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df["anomaly"] = iso.fit_predict(X)
# -1 = anomaly, 1 = normal

anomalies = df[df["anomaly"] == -1]

anomaly_output = DATA_DIR / "hotspot_anomalies.csv"
anomalies.to_csv(anomaly_output, index=False)

print("Isolation Forest complete")
print("Number of anomalous files:", len(anomalies))
print("All plots saved in figures/")

