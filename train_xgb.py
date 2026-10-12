"""
XGBoost pairwise reward model for recourse preferences.

Run:  python train_xgb.py --study study1                      (mock data)
      python train_xgb.py --data /path/to/real --study study2  (real data, secure machine only)

Expects <data>/<study>_preference_train.csv and <data>/<study>_preference_test.csv,
or pass --train / --test with exact file paths.
Prints ONLY summary numbers, never rows, so the output is safe to share.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# !! CHECK WITH FIRDAUS: what does user_choice mean? Assumed: 0 = picked cf1, 1 = picked cf2.
# If it is the other way round, run with --flip. Getting it wrong turns accuracy into (1 - accuracy).
OPTS = ["cf1_feats_values", "cf2_feats_values"]
COLS = ["pid", "scenario_num", "original"] + OPTS + ["user_choice"]


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}. Found {list(df.columns)}")
    for c in ["original"] + OPTS:
        try:
            df[c] = df[c].apply(json.loads)
        except Exception as e:
            sys.exit(f"{path}: could not parse JSON in column '{c}' ({type(e).__name__}). Check the cell format.")
    bad = set(df["user_choice"].unique()) - {0, 1}
    if bad:
        sys.exit(f"{path}: user_choice has values other than 0/1: {sorted(bad)}. Update the label mapping.")
    df["user_choice"] = df["user_choice"].astype(int)
    keys = set(df["original"].iloc[0])
    for c in ["original"] + OPTS:
        if any(set(d) != keys for d in df[c]):
            sys.exit(f"{path}: column '{c}' has rows with different feature names.")
    return df


def to_num(values):
    """Parse numbers, tolerating '108480' strings, thousands commas, '$' and stray spaces."""
    s = pd.Series(list(values), dtype=object).map(lambda v: v if v is None else str(v))
    s = s.str.replace(r"[,$\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def split_types(df):
    """Numeric = at least 98% of values parse as numbers. Rest = categorical."""
    num, cat = [], []
    for k in df["original"].iloc[0]:
        vals = to_num(d.get(k) for col in ["original"] + OPTS for d in df[col])
        (num if vals.notna().mean() >= 0.98 else cat).append(k)
    return num, cat


def drop_unreadable(df, num, name):
    """Drop choices where any numeric feature in any of the 3 columns can't be read as a number."""
    bad = np.zeros(len(df), dtype=bool)
    for c in ["original"] + OPTS:
        for f in num:
            bad |= to_num(d.get(f) for d in df[c]).isna().values
    if bad.any():
        print(f"  note: dropped {bad.sum()} of {len(df)} {name} choices with unreadable numbers")
    return df[~bad].reset_index(drop=True)


def option_features(df, opt, num, cat, scale, cats):
    """One row of features per (scenario, option): context + what changed + how much."""
    o = pd.DataFrame(list(df["original"]), index=df.index)
    c = pd.DataFrame(list(df[opt]), index=df.index)
    X = pd.DataFrame(index=df.index)
    l1 = pd.Series(0.0, index=df.index)
    for f in num:
        ov, cv = to_num(o[f]).set_axis(df.index), to_num(c[f]).set_axis(df.index)
        d = (cv - ov) / scale[f]
        X[f"orig_{f}"] = ov
        X[f"delta_{f}"] = d
        X[f"changed_{f}"] = (cv != ov).astype(int)
        l1 = l1 + d.abs().fillna(0)
    for f in cat:
        ov, cv = o[f].astype(str).str.strip(), c[f].astype(str).str.strip()
        X[f"changed_{f}"] = (cv != ov).astype(int)
        for v in cats[f]:
            X[f"orig_{f}={v}"] = (ov == v).astype(int)
            X[f"new_{f}={v}"] = (cv == v).astype(int)
    X["n_changed"] = X[[k for k in X if k.startswith("changed_")]].sum(axis=1)
    X["l1_norm"] = l1
    return X


def build(df, num, cat, scale, cats, flip):
    X1 = option_features(df, OPTS[0], num, cat, scale, cats)
    X2 = option_features(df, OPTS[1], num, cat, scale, cats)
    picked_cf2 = (df["user_choice"].values == (0 if flip else 1)).astype(int)
    return X1, X2, picked_cf2


def to_ranking(X1, X2, picked_cf2):
    """Interleave rows: [s0 cf1, s0 cf2, s1 cf1, s1 cf2, ...]; each scenario is one query group of 2."""
    n = len(X1)
    X = pd.concat([X1, X2]).reset_index(drop=True)
    y = np.concatenate([1 - picked_cf2, picked_cf2])
    qid = np.concatenate([np.arange(n), np.arange(n)])
    order = np.argsort(qid, kind="stable")
    return X.iloc[order].reset_index(drop=True), y[order], qid[order]


def report(name, y, pred):
    print(f"  {name:<22} acc={accuracy_score(y, pred):.3f}  "
          f"f1={f1_score(y, pred, zero_division=0):.3f}  "
          f"prec={precision_score(y, pred, zero_division=0):.3f}  "
          f"rec={recall_score(y, pred, zero_division=0):.3f}")


def fit_predict(tr, te, num, cat, flip, seed):
    """Fit on tr, score te. Scaling and category lists come from tr only (no test leakage)."""
    scale = {}
    for f in num:
        sd = to_num(d[f] for d in tr["original"]).std()
        scale[f] = sd if np.isfinite(sd) and sd > 0 else 1.0
    cats = {f: sorted({str(d[f]).strip() for col in ["original"] + OPTS for d in tr[col]}) for f in cat}

    X1, X2, ytr = build(tr, num, cat, scale, cats, flip)
    Xtr, rtr, qtr = to_ranking(X1, X2, ytr)
    T1, T2, yte = build(te, num, cat, scale, cats, flip)

    model = XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed)
    model.fit(Xtr, rtr, qid=qtr)
    # Reward = model score. Predict cf2 if its reward is higher (ties -> cf1).
    pred = (model.predict(T2[Xtr.columns]) > model.predict(T1[Xtr.columns])).astype(int)
    prox = (T2["l1_norm"].values < T1["l1_norm"].values).astype(int)
    train_pred = (model.predict(X2[Xtr.columns]) > model.predict(X1[Xtr.columns])).astype(int)
    return dict(y=yte, pred=pred, prox=prox, train_acc=accuracy_score(ytr, train_pred),
                majority=np.full_like(yte, int(ytr.mean() > 0.5)))


def scenario_key(df):
    """Same loan + same two options = same scenario, whichever side each option was shown on."""
    j = lambda d: json.dumps(d, sort_keys=True)
    return [j(o) + "|" + "|".join(sorted([j(a), j(b)]))
            for o, a, b in zip(df["original"], df[OPTS[0]], df[OPTS[1]])]


def ci(acc, n):
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)
    return f"{acc:.3f} (95% CI {acc - half:.3f}-{acc + half:.3f})"


def run_cv(tr, te, num, cat, a):
    from sklearn.model_selection import GroupKFold
    df = pd.concat([tr, te]).reset_index(drop=True)
    keys = pd.Series(scenario_key(df))
    seen = set(scenario_key(tr))
    print(f"\n  Scenario overlap: {keys.nunique()} unique scenarios in {len(df)} choices; "
          f"{np.mean([k in seen for k in scenario_key(te)]):.1%} of the original test choices "
          f"use a scenario that is also in train")
    for label, groups in [("unseen PEOPLE", df["pid"].values), ("unseen SCENARIOS", keys.factorize()[0])]:
        k = min(a.folds, len(np.unique(groups)))
        res = {"XGBoost": [], "proximity": [], "majority side": [], "train": []}
        for i, (itr, ite) in enumerate(GroupKFold(n_splits=k).split(df, groups=groups)):
            r = fit_predict(df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True),
                            num, cat, a.flip, a.seed + i)
            res["XGBoost"].append(accuracy_score(r["y"], r["pred"]))
            res["proximity"].append(accuracy_score(r["y"], r["prox"]))
            res["majority side"].append(accuracy_score(r["y"], r["majority"]))
            res["train"].append(r["train_acc"])
        print(f"  {k}-fold CV, test on {label}:")
        for name in ["majority side", "proximity", "XGBoost"]:
            v = np.array(res[name])
            print(f"    {name:<14} mean acc={v.mean():.3f}  (folds range {v.min():.3f}-{v.max():.3f})")
        print(f"    XGBoost train acc={np.mean(res['train']):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mock_data")
    ap.add_argument("--study", default="study1", choices=["study1", "study2"])
    ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--flip", action="store_true", help="use if user_choice 1 means cf1")
    ap.add_argument("--cv", action="store_true", help="also run the unseen-people and unseen-scenario checks")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    tr = load(a.train or f"{a.data}/{a.study}_preference_train.csv")
    te = load(a.test or f"{a.data}/{a.study}_preference_test.csv")
    num, cat = split_types(tr)
    num_te, cat_te = split_types(te)
    if set(num) != set(num_te) or set(cat) != set(cat_te):
        print("Train and test files disagree. Details (names and counts only, no values):")
        print(f"  train numeric={sorted(num)} categorical={sorted(cat)}")
        print(f"  test  numeric={sorted(num_te)} categorical={sorted(cat_te)}")
        for f in sorted(set(num) ^ set(num_te)):
            for name, df in [("train", tr), ("test", te)]:
                n_bad = sum(to_num(d.get(f) for d in df[c]).isna().sum()
                            for c in ["original"] + OPTS)
                print(f"  '{f}' in {name}: {n_bad} of {3 * len(df)} cells are not numbers")
        sys.exit(1)
    tr, te = drop_unreadable(tr, num, "train"), drop_unreadable(te, num, "test")

    r = fit_predict(tr, te, num, cat, a.flip, a.seed)
    yte, n = r["y"], len(r["y"])
    shared = len(set(tr["pid"]) & set(te["pid"]))
    print(f"{a.study}: train {len(tr)} choices ({tr['pid'].nunique()} people), "
          f"test {len(te)} choices ({te['pid'].nunique()} people, {shared} also in train)")
    print(f"  features: numeric={num} categorical={cat}")
    print(f"  test share where cf2 was picked: {yte.mean():.3f}")
    report("always pick cf1", yte, np.zeros_like(yte))
    report("proximity (smaller L1)", yte, r["prox"])
    report("XGBoost ranker", yte, r["pred"])
    print(f"  XGBoost test acc {ci(accuracy_score(yte, r['pred']), n)}")
    print(f"  XGBoost train accuracy: {r['train_acc']:.3f}  (much higher than test = overfitting)")
    print("  (positive class for f1/prec/rec = 'participant picked cf2')")
    if accuracy_score(yte, r["prox"]) < 0.45:
        print("  WARNING: proximity is well below 50%. user_choice may be coded the other way; try --flip.")
    if a.cv:
        run_cv(tr, te, num, cat, a)
    return accuracy_score(yte, r["pred"]), accuracy_score(yte, r["prox"])


if __name__ == "__main__":
    main()
