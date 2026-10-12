"""
XGBoost pairwise reward model for recourse preferences.

Run:  python train_xgb.py --data mock_data --study study1
      python train_xgb.py --data /path/to/real/folder --study study2

Expects <data>/<study>_preference_train.csv and <data>/<study>_preference_test.csv.
Prints ONLY summary numbers, never rows, so the output is safe to share.
"""
import argparse, json
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# !! CHECK WITH FIRDAUS: what does user_choice mean? This script assumes 0 = picked cf1, 1 = picked cf2.
# If it is the other way round, set this to False. Getting it wrong flips accuracy to (1 - accuracy).
CHOICE_1_MEANS_CF2 = True

OPTS = ["cf1_feats_values", "cf2_feats_values"]


def load(path):
    df = pd.read_csv(path, sep=None, engine="python")          # handles tab or comma
    for c in ["original"] + OPTS:
        df[c] = df[c].apply(json.loads)
    df["user_choice"] = df["user_choice"].astype(int)
    return df


def split_types(df):
    """Numeric = every value parses as a number (handles '108480' strings). Rest = categorical."""
    keys = list(df["original"].iloc[0].keys())
    num, cat = [], []
    for k in keys:
        vals = pd.to_numeric(pd.Series([d[k] for d in df["original"]]), errors="coerce")
        (num if vals.notna().all() else cat).append(k)
    return num, cat


def option_features(df, opt, num, cat, scale, cats):
    """One row of features per (scenario, option): context + what changed + how much."""
    o = pd.DataFrame(list(df["original"]))
    c = pd.DataFrame(list(df[opt]))
    X = pd.DataFrame(index=df.index)
    l1 = 0
    for f in num:
        ov, cv = pd.to_numeric(o[f]), pd.to_numeric(c[f])
        d = (cv - ov) / scale[f]
        X[f"orig_{f}"] = ov
        X[f"delta_{f}"] = d
        X[f"changed_{f}"] = (cv != ov).astype(int)
        l1 = l1 + d.abs()
    for f in cat:
        ov, cv = o[f].astype(str).str.strip(), c[f].astype(str).str.strip()
        X[f"changed_{f}"] = (cv != ov).astype(int)
        for v in cats[f]:
            X[f"orig_{f}={v}"] = (ov == v).astype(int)
            X[f"new_{f}={v}"] = (cv == v).astype(int)
    X["n_changed"] = X[[k for k in X if k.startswith("changed_")]].sum(axis=1)
    X["l1_norm"] = l1
    return X


def to_ranking(df, num, cat, scale, cats):
    """Stack cf1 and cf2 rows; each scenario is one query group of 2."""
    X1 = option_features(df, OPTS[0], num, cat, scale, cats)
    X2 = option_features(df, OPTS[1], num, cat, scale, cats)
    picked_cf2 = df["user_choice"] == (1 if CHOICE_1_MEANS_CF2 else 0)
    n = len(df)
    X = pd.concat([X1, X2]).reset_index(drop=True)
    y = np.concatenate([(~picked_cf2).astype(int), picked_cf2.astype(int)])
    qid = np.concatenate([np.arange(n), np.arange(n)])
    order = np.argsort(qid, kind="stable")                    # XGBRanker needs groups contiguous
    return X.iloc[order].reset_index(drop=True), y[order], qid[order], X1, X2, picked_cf2.astype(int).values


def report(name, y_true_cf2, pred_cf2):
    print(f"  {name:<22} acc={accuracy_score(y_true_cf2, pred_cf2):.3f}  "
          f"f1={f1_score(y_true_cf2, pred_cf2):.3f}  "
          f"prec={precision_score(y_true_cf2, pred_cf2, zero_division=0):.3f}  "
          f"rec={recall_score(y_true_cf2, pred_cf2):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mock_data")
    ap.add_argument("--study", default="study1", choices=["study1", "study2"])
    a = ap.parse_args()

    tr = load(f"{a.data}/{a.study}_preference_train.csv")
    te = load(f"{a.data}/{a.study}_preference_test.csv")
    num, cat = split_types(tr)

    # Fit scaling and category lists on TRAIN only
    o = pd.DataFrame(list(tr["original"]))
    scale = {f: pd.to_numeric(o[f]).std() or 1.0 for f in num}
    cats = {f: sorted(set(o[f].astype(str).str.strip())
                      | set(pd.DataFrame(list(tr[OPTS[0]]))[f].astype(str).str.strip())
                      | set(pd.DataFrame(list(tr[OPTS[1]]))[f].astype(str).str.strip())) for f in cat}

    Xtr, ytr, qtr, *_ = to_ranking(tr, num, cat, scale, cats)
    _, _, _, T1, T2, yte = to_ranking(te, num, cat, scale, cats)

    model = XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0)
    model.fit(Xtr, ytr, qid=qtr)

    # Reward = model score. Predict cf2 if its reward is higher.
    T1, T2 = T1[Xtr.columns], T2[Xtr.columns]
    pred = (model.predict(T2) > model.predict(T1)).astype(int)
    prox = (T2["l1_norm"] < T1["l1_norm"]).astype(int).values

    print(f"{a.study}: train {len(tr)} choices, test {len(te)} choices, "
          f"{tr['pid'].nunique()} train participants")
    print(f"  features: numeric={num} categorical={cat}")
    print(f"  test share where cf2 was picked: {yte.mean():.3f}  (majority-class baseline = "
          f"{max(yte.mean(), 1 - yte.mean()):.3f})")
    report("always pick cf1", yte, np.zeros_like(yte))
    report("proximity (smaller L1)", yte, prox)
    report("XGBoost ranker", yte, pred)
    print("  (positive class for f1/prec/rec = 'participant picked cf2')")


if __name__ == "__main__":
    main()
