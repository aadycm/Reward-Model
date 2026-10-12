"""
Sanity tests for the pipeline. Uses only generated data.

Run:  python test_pipeline.py
"""
import contextlib, io, json, os, tempfile
import numpy as np
import pandas as pd
import make_mock_data as mm
import train_xgb as tx


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def write_pair(folder, study, rows_train, rows_test, spec, string_values):
    mm.write_train(f"{folder}/{study}_preference_train.csv", rows_train, spec, as_string=string_values)
    mm.write_test(f"{folder}/{study}_preference_test.csv", rows_test, spec)


def l1(o, cf, spec):
    return sum(abs(cf[f] - o[f]) / (spec["numeric"][f][1] - spec["numeric"][f][0]) for f in spec["numeric"])


def oracle_rows(spec, n, rng, noise=0.0):
    """Choice = whichever option is closer in L1 (optionally with label noise)."""
    rows = []
    for i in range(n):
        o = mm.random_original(spec, rng)
        a, b = mm.random_cf(o, spec, rng), mm.random_cf(o, spec, rng)
        choice = int(l1(o, b, spec) < l1(o, a, spec))
        if rng.random() < noise:
            choice = 1 - choice
        rows.append((i // 10 + 1, i % 10 + 1, o, a, b, choice))
    return rows


def test_formats_parse():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1)
        for study, spec, s in [("study1", mm.STUDY1, False), ("study2", mm.STUDY2, True)]:
            tr, te = mm.make_study(spec, 5, 8, 2, rng)
            write_pair(d, study, tr, te, spec, s)
            a, b = tx.load(f"{d}/{study}_preference_train.csv"), tx.load(f"{d}/{study}_preference_test.csv")
            assert len(a) == 40 and len(b) == 10
            num, cat = tx.split_types(a)
            assert set(num) == set(spec["numeric"]) and set(cat) == set(spec["categorical"]), (num, cat)
            assert tx.split_types(b) == (num, cat)
            # test file really is tab-separated and unquoted, like the real one
            line = open(f"{d}/{study}_preference_test.csv").readlines()[1]
            assert line.count("\t") == 5 and '""' not in line
            # train file is comma-separated with doubled quotes
            assert '""' in open(f"{d}/{study}_preference_train.csv").readlines()[1]
    print("ok  formats parse; numeric/categorical detected (incl. study2 numbers stored as strings)")


def test_grouping():
    X1 = pd.DataFrame({"v": [10, 20, 30]}); X2 = pd.DataFrame({"v": [11, 21, 31]})
    X, y, q = tx.to_ranking(X1, X2, np.array([1, 0, 1]))
    assert list(X["v"]) == [10, 11, 20, 21, 30, 31]
    assert list(q) == [0, 0, 1, 1, 2, 2]
    assert list(y) == [0, 1, 1, 0, 0, 1]          # exactly one chosen option per group
    print("ok  each choice becomes 2 rows in one group, chosen = 1")


def test_learns_known_rule():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(2)
        write_pair(d, "study1", oracle_rows(mm.STUDY1, 3000, rng), oracle_rows(mm.STUDY1, 800, rng),
                   mm.STUDY1, False)
        xgb, prox = quiet(tx.main, ["--data", d, "--study", "study1"])
    assert prox == 1.0, prox                       # proximity is the true rule here
    assert xgb > 0.9, xgb                          # model should recover it
    print(f"ok  recovers a known rule: XGBoost {xgb:.3f}, proximity {prox:.3f}")


def test_flip_mirrors():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(3)
        write_pair(d, "study1", oracle_rows(mm.STUDY1, 1000, rng, 0.2), oracle_rows(mm.STUDY1, 400, rng, 0.2),
                   mm.STUDY1, False)
        x0, p0 = quiet(tx.main, ["--data", d, "--study", "study1"])
        x1, p1 = quiet(tx.main, ["--data", d, "--study", "study1", "--flip"])
    assert abs(p0 + p1 - 1) < 1e-9                 # flipping labels mirrors the fixed baseline exactly
    assert abs(x0 - x1) < 0.03, (x0, x1)           # XGBoost relearns the flipped rule, so it can't reveal it
    assert p1 < 0.45                               # ...but the proximity warning fires on the wrong mapping
    print(f"ok  wrong label mapping shows up in proximity ({p0:.3f} vs {p1:.3f}), not in XGBoost")


def test_no_signal_is_chance():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(4)
        tr = [r[:5] + (int(rng.random() < 0.5),) for r in oracle_rows(mm.STUDY2, 2000, rng)]
        te = [r[:5] + (int(rng.random() < 0.5),) for r in oracle_rows(mm.STUDY2, 2000, rng)]
        write_pair(d, "study2", tr, te, mm.STUDY2, True)
        xgb, _ = quiet(tx.main, ["--data", d, "--study", "study2"])
    assert 0.45 < xgb < 0.55, xgb                  # no leakage: random labels give ~50%
    print(f"ok  random labels give chance accuracy ({xgb:.3f}), so there is no test leakage")


def test_bad_labels_rejected():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(5)
        rows = [r[:5] + (2,) for r in oracle_rows(mm.STUDY1, 20, rng)]
        mm.write_test(f"{d}/x.csv", rows, mm.STUDY1)
        try:
            quiet(tx.load, f"{d}/x.csv"); raise AssertionError("should have failed")
        except SystemExit as e:
            assert "other than 0/1" in str(e)
    print("ok  user_choice values other than 0/1 stop the run with a clear message")


def test_messy_numbers():
    """A few odd loan_amount cells in train only (like the real study1 file)."""
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(6)
        tr = oracle_rows(mm.STUDY1, 400, rng)
        odd = {0: "12,345.50", 1: "", 2: "N/A"}         # first is fixable, other two are not
        tr = [(p, s, {**o, "loan_amount": odd[i]} if i in odd else o, a, b, c)
              for i, (p, s, o, a, b, c) in enumerate(tr)]
        write_pair(d, "study1", tr, oracle_rows(mm.STUDY1, 100, rng), mm.STUDY1, False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            xgb, prox = tx.main(["--data", d, "--study", "study1"])
        text = out.getvalue()
    assert "numeric=['income', 'credit_score', 'loan_amount']" in text, text
    assert "dropped 2 of 400 train choices" in text, text
    assert prox == 1.0
    print("ok  odd number cells: '12,345.50' is read, blanks/'N/A' rows are dropped and counted")


def test_cv_catches_memorisation():
    """Choices depend only on WHICH scenario it is (no rule), and scenarios repeat across people.
    The fixed split looks great; testing on unseen scenarios must fall to ~chance."""
    import re
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(7)
        pool = []
        for _ in range(60):
            o = mm.random_original(mm.STUDY1, rng)
            pool.append((o, mm.random_cf(o, mm.STUDY1, rng), mm.random_cf(o, mm.STUDY1, rng), int(rng.random() < 0.5)))
        def rows(people, per):
            out = []
            for p in people:
                for s, i in enumerate(rng.choice(len(pool), per, replace=False), 1):
                    o, a, b, c = pool[i]
                    out.append((p, s, o, a, b, c if rng.random() < 0.9 else 1 - c))
            return out
        write_pair(d, "study1", rows(range(1, 81), 40), rows(range(1, 81), 10), mm.STUDY1, False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fixed, _ = tx.main(["--data", d, "--study", "study1", "--cv"])
        text = out.getvalue()
    unseen = float(re.search(r"unseen SCENARIOS:.*?XGBoost +mean acc=([0-9.]+)", text, re.S).group(1))
    assert fixed > 0.8, fixed
    assert unseen < 0.62, unseen
    print(f"ok  memorisation exposed: fixed split {fixed:.3f}, unseen scenarios {unseen:.3f}")


if __name__ == "__main__":
    for t in [test_formats_parse, test_grouping, test_learns_known_rule, test_flip_mirrors,
              test_no_signal_is_chance, test_bad_labels_rejected, test_messy_numbers, test_cv_catches_memorisation]:
        t()
    print("\nAll tests passed.")
