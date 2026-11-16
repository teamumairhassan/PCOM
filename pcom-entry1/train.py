import numpy as np, pandas as pd, joblib, warnings
from google.colab import files
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    BACKEND = "sklearn"
print("Backend:", BACKEND)

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

def make_clf(w):
    if BACKEND == "xgboost":
        return xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, min_child_weight=3,
                                 subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                                 scale_pos_weight=w, random_state=42, n_jobs=-1, tree_method="hist")
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=10, l2_regularization=2.0,
                                          class_weight={0: 1.0, 1: w}, random_state=42)

def make_reg():
    if BACKEND == "xgboost":
        return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, min_child_weight=2,
                                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                                random_state=42, n_jobs=-1, tree_method="hist")
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=5, l2_regularization=2.0, random_state=42)

# ---------- upload training CSV (updated.csv) ----------
up = files.upload()
df = pd.read_csv(list(up.keys())[0])
print("Loaded", df.shape)

ref = pd.to_datetime(df["published_at"], utc=True, format="mixed").max()
X = build_features(df, ref)
groups = df["purl"].str.extract(r"pkg:npm/(.+)@")[0]

bundle = dict(ref_date=str(ref), columns=list(X.columns), models={})
for s in SEV:
    cnt = df[f"target_ghsa_{s}_count_90d"].fillna(0).astype(int)
    y = (cnt > 0).astype(int)
    w = min((len(y) - y.sum()) / max(y.sum(), 1), 10.0)

    # out-of-fold predictions (grouped by package) to choose the threshold
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof[va] = make_clf(w).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    ts = np.linspace(0.05, 0.95, 91)
    thr = float(ts[np.argmax([f1_score(y, oof >= t, zero_division=0) for t in ts])])
    if accuracy_score(y, oof >= thr) < 1 - y.mean():      # weak signal -> fall back to accuracy
        ts2 = np.linspace(0.05, 0.99, 95)
        thr = float(ts2[np.argmax([accuracy_score(y, oof >= t) for t in ts2])])
    print(f"[{s}] threshold={thr:.2f}  OOF acc={accuracy_score(y, oof >= thr):.3f}  (always-0: {1 - y.mean():.3f})")

    clf = make_clf(w).fit(X, y)                            # final model on ALL data
    pos = cnt > 0
    reg = make_reg().fit(X[pos], np.log1p(cnt[pos])) if pos.sum() >= 20 else None
    bundle["models"][s] = dict(clf=clf, reg=reg, thr=thr)

joblib.dump(bundle, "model.joblib", compress=3)
print("Saved model.joblib")
files.download("model.joblib")