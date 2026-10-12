"""
Make MOCK versions of the four preference files, in the same formats as the real ones.
No real data is used: every value is random. A hidden rule ("prefer small, simple changes",
plus per-person noise) decides user_choice, so a model has something to learn.

Run:
    python make_mock_data.py                      # default size
    python make_mock_data.py --people 500         # bigger
    python make_mock_data.py --out mock_data --seed 1

Formats copied from the real files:
    *_train.csv : comma-separated, JSON cells quoted with doubled quotes, ", " between items
    *_test.csv  : tab-separated, JSON cells unquoted, "," between items
    study2 train stores numbers as strings ("108480"); study2 test stores them as floats (57630.0)
    study2 categories have a leading space (" Graduate", " Yes")
"""
import argparse, csv, json, os
import numpy as np

# Category lists are guesses from example rows. Check them against the paper.
STUDY1 = {
    "order": ["income", "credit_score", "loan_amount", "employment_type", "education"],
    "numeric": {"income": (20000, 400000, 2), "credit_score": (400, 850, 0), "loan_amount": (1000, 30000, 2)},
    "categorical": {
        "employment_type": ["Self", "Private", "Government"],
        "education": ["Less than High school", "High school", "Associate's", "Bachelor's", "Master's"],
    },
}
STUDY2 = {
    "order": ["income_annum", "loan_amount", "loan_term", "education", "self_employed", "credit_score"],
    "numeric": {"income_annum": (10000, 120000, -1), "loan_amount": (40000, 400000, -1),
                "loan_term": (2, 20, 0), "credit_score": (400, 600, 0)},
    "categorical": {"education": [" Graduate", " Not Graduate"], "self_employed": [" Yes", " No"]},
}
LOWER_IS_BETTER = {"loan_amount", "loan_term"}


def rnd(x, dec):
    return float(round(x, dec))


def random_original(spec, rng):
    row = {}
    for f in spec["order"]:
        if f in spec["numeric"]:
            lo, hi, dec = spec["numeric"][f]
            row[f] = rnd(rng.uniform(lo, hi), dec)
        else:
            row[f] = str(rng.choice(spec["categorical"][f]))
    return row


def random_cf(orig, spec, rng):
    """Change 1-3 features, numeric ones in the direction that helps approval."""
    cf = dict(orig)
    k = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
    for f in rng.choice(spec["order"], size=k, replace=False):
        if f in spec["numeric"]:
            lo, hi, dec = spec["numeric"][f]
            step = rng.uniform(0.02, 0.5) * (hi - lo)
            sign = -1 if f in LOWER_IS_BETTER else 1
            new = rnd(np.clip(orig[f] + sign * step, lo, hi), dec)
            cf[f] = new if new != orig[f] else rnd(orig[f] + sign * (hi - lo) * 0.05, dec)
        else:
            cf[f] = str(rng.choice([v for v in spec["categorical"][f] if v != orig[f]]))
    return cf


def cost(orig, cf, spec, w):
    """Hidden rule: weighted normalised L1 distance + penalty per changed feature."""
    c = 0.0
    for f in spec["order"]:
        if f in spec["numeric"]:
            lo, hi, _ = spec["numeric"][f]
            c += w[f] * abs(cf[f] - orig[f]) / (hi - lo)
        if cf[f] != orig[f]:
            c += 0.3
    return c


def make_study(spec, n_people, n_train, n_test, rng):
    train, test = [], []
    for pid in range(1, n_people + 1):
        # each person weights features a bit differently (makes personalisation matter)
        w = {f: float(rng.lognormal(0, 0.4)) for f in spec["numeric"]}
        for s in range(1, n_train + n_test + 1):
            o = random_original(spec, rng)
            a, b = random_cf(o, spec, rng), random_cf(o, spec, rng)
            while b == a:
                b = random_cf(o, spec, rng)
            gap = cost(o, b, spec, w) - cost(o, a, spec, w)
            p_cf1 = 1 / (1 + np.exp(-4 * gap))
            choice = 0 if rng.random() < p_cf1 else 1      # ASSUMPTION: 0 = picked cf1, 1 = picked cf2
            rec = (pid, s if s <= n_train else s - n_train, o, a, b, choice)
            (train if s <= n_train else test).append(rec)
    return train, test


def fmt_num(v, dec, as_string):
    if as_string:
        return str(int(v)) if dec <= 0 else str(v)
    return v


def encode(d, spec, as_string):
    return {k: fmt_num(v, spec["numeric"][k][2], as_string) if k in spec["numeric"] else v for k, v in d.items()}


def write_train(path, rows, spec, as_string):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "scenario_num", "original", "cf1_feats_values", "cf2_feats_values", "user_choice"])
        for pid, s, o, a, b, c in rows:
            w.writerow([pid, s] + [json.dumps(encode(x, spec, as_string)) for x in (o, a, b)] + [c])


def write_test(path, rows, spec):
    with open(path, "w") as fh:
        fh.write("\t".join(["pid", "scenario_num", "original", "cf1_feats_values", "cf2_feats_values",
                            "user_choice"]) + "\n")
        for pid, s, o, a, b, c in rows:
            cells = [json.dumps(encode(x, spec, False), separators=(",", ": ")) for x in (o, a, b)]
            fh.write("\t".join([str(pid), str(s)] + cells + [str(c)]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock_data")
    ap.add_argument("--people", type=int, default=200, help="participants per study")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # 80/20 split per participant, as in Firdaus's description
    tr, te = make_study(STUDY1, a.people, 40, 10, rng)
    write_train(f"{a.out}/study1_preference_train.csv", tr, STUDY1, as_string=False)
    write_test(f"{a.out}/study1_preference_test.csv", te, STUDY1)
    print(f"study1: {len(tr)} train, {len(te)} test")

    tr, te = make_study(STUDY2, a.people, 32, 8, rng)
    write_train(f"{a.out}/study2_preference_train.csv", tr, STUDY2, as_string=True)
    write_test(f"{a.out}/study2_preference_test.csv", te, STUDY2)
    print(f"study2: {len(tr)} train, {len(te)} test")
