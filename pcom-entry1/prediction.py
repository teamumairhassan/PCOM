import numpy as np, pandas as pd, joblib, time, warnings
from google.colab import files
warnings.filterwarnings("ignore")

SEV = ["low", "moderate", "high", "critical"]

def mc(df, p):
    return df[[f"{p}_M{i}" for i in range(12)]].astype(float)

def build_features(df, ref):
    X = pd.DataFrame(index=df.index)
    static = ["unpacked_size","file_count","dependency_count","optional_dependency_count",
              "peer_dependency_count","dev_dependency_count","bundled_dependency_count",
              "ghsa_low_count","ghsa_moderate_count","ghsa_high_count",
              "ghsa_critical_count","ghsa_unknown_count"]
    for c in static:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["ghsa_total_count"] = X[["ghsa_low_count","ghsa_moderate_count","ghsa_high_count",
                               "ghsa_critical_count","ghsa_unknown_count"]].sum(axis=1)
    created = pd.to_datetime(df["package_created_at"], utc=True, format="mixed")
    pub = pd.to_datetime(df["published_at"], utc=True, format="mixed")
    ref = pd.Timestamp(ref)
    X["package_age_days"] = (ref - created).dt.days
    X["version_age_days"] = (ref - pub).dt.days
    X["version_lag_days"] = (pub - created).dt.days
    X["published_month"] = pub.dt.month
    name = df["purl"].str.extract(r"pkg:npm/(.+)@")[0]
    X["is_scoped"] = name.str.startswith("@").astype(int)
    ver = df["purl"].str.extract(r"@([^@]+)$")[0]
    X["major_version"] = pd.to_numeric(ver.str.extract(r"^(\d+)")[0], errors="coerce")
    X["is_prerelease"] = ver.str.contains("-", regex=False).astype(int)
    series = ["package_downloads","package_stable_release_count","issues_opened","issues_closed",
              "issues_reopened","pull_requests_opened","pull_requests_closed_unmerged",
              "pull_requests_merged","pull_requests_reopened","issues_open_at_month_end",
              "pull_requests_open_at_month_end","ghsa_low_count","ghsa_moderate_count",
              "ghsa_high_count","ghsa_critical_count"]
    new = {}
    for s in series:
        m = mc(df, s)
        for c in m.columns: new[c] = m[c]
        new[f"{s}_sum"] = m.sum(axis=1, min_count=1)
        new[f"{s}_mean"] = m.mean(axis=1)
        l3 = m[[f"{s}_M{i}" for i in range(3)]].sum(axis=1, min_count=1)
        p3 = m[[f"{s}_M{i}" for i in range(3, 6)]].sum(axis=1, min_count=1)
        new[f"{s}_last3_sum"], new[f"{s}_prev3_sum"], new[f"{s}_trend"] = l3, p3, l3 - p3
    X = pd.concat([X, pd.DataFrame(new)], axis=1)
    g12 = sum(mc(df, f"ghsa_{s}_count") for s in SEV)
    X["ghsa_12m_total"] = g12.sum(axis=1)
    X["ghsa_months_with_advisory"] = (g12 > 0).sum(axis=1)
    X["ghsa_last3m_total"] = g12[[f"ghsa_low_count_M{i}" for i in range(3)]].sum(axis=1)
    X["issue_close_ratio"] = X["issues_closed_sum"] / (X["issues_opened_sum"] + 1)
    X["pr_merge_ratio"] = X["pull_requests_merged_sum"] / (X["pull_requests_opened_sum"] + 1)
    X["downloads_per_release"] = X["package_downloads_sum"] / (X["package_stable_release_count_sum"] + 1)
    X["download_days_observed_total"] = mc(df, "package_download_days_observed").sum(axis=1)
    return X

# ---------- upload model.joblib (skip if already in the Colab folder) ----------
import os
if not os.path.exists("model.joblib"):
    print("Upload model.joblib")
    files.upload()
bundle = joblib.load("model.joblib")

# ---------- upload the input CSV ----------
print("Upload the input CSV")
up = files.upload()
inp = [k for k in up if k.lower().endswith(".csv")][0]

t0 = time.time()
df = pd.read_csv(inp)
X = build_features(df, bundle["ref_date"]).reindex(columns=bundle["columns"])

out = pd.DataFrame({"purl": df["purl"]})
for s in SEV:
    m = bundle["models"][s]
    p = m["clf"].predict_proba(X)[:, 1]
    flag = p >= m["thr"]
    count = np.zeros(len(df), dtype=int)
    if flag.any():
        if m["reg"] is not None:
            count[flag] = np.maximum(np.rint(np.expm1(m["reg"].predict(X[flag]))).astype(int), 1)
        else:
            count[flag] = 1
    out[f"target_ghsa_{s}_count_90d"] = count
out["target_ghsa_unknown_count_90d"] = 0

out.to_csv("predictions.csv", index=False)
print(f"Done in {time.time() - t0:.1f}s, {len(out)} rows")
display(out)
files.download("predictions.csv")