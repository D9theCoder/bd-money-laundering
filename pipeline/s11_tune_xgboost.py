"""Hyperparameter search for the XGBoost models (X1: base features, X2: base + graph features).

Protocol (no test data is touched): random search with a fixed seed. Every trial is fitted twice on a rolling,
time ordered scheme that stays inside the training period plus the validation period:
  fold A: train 1 to 6 Sept -> validate 7 to 8 Sept (the official validation split)
  fold B: train 1 to 4 Sept -> validate 5 to 6 Sept
The selection score is the mean validation PR-AUC (average precision) of the two folds, so one lucky two-day
window cannot decide the winner. The best configuration is written to logs/tuning and later copied to
config.yaml; stage train_xgb then retrains it once and evaluates the test periods.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from common import CFG, ROOT, log, utcnow
from s05_train_xgboost import LOCAL, fetch_split, read_json_hdfs

L = log("tune_xgboost")
MC = CFG["models"]
N_TRIALS = int(os.environ.get("TUNE_TRIALS", 40))
N_JOBS = int(os.environ.get("TUNE_JOBS", 8))
OUT = "/opt/aml/logs/tuning"
FOLDS = {"A": (("2022-09-01", "2022-09-06"), ("2022-09-07", "2022-09-08")),
         "B": (("2022-09-01", "2022-09-04"), ("2022-09-05", "2022-09-06"))}


def sample(rng):
    # Learning rate, lambda and alpha are drawn log-uniformly because useful values span orders of magnitude.
    # spw_factor scales the balanced scale_pos_weight (1.0 = fully balanced); small factors down-weight positives.
    return {
        "n_estimators": int(rng.choice([100, 200, 300, 500])),
        "learning_rate": round(float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))), 4),
        "max_depth": int(rng.integers(3, 11)),
        "min_child_weight": int(rng.choice([1, 5, 20, 50])),
        "subsample": round(float(rng.uniform(0.6, 1.0)), 2),
        "colsample_bytree": round(float(rng.uniform(0.5, 1.0)), 2),
        "reg_lambda": round(float(np.exp(rng.uniform(np.log(0.1), np.log(20)))), 3),
        "reg_alpha": round(float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))), 4),
        "gamma": float(rng.choice([0.0, 0.5, 2.0])),
        "spw_factor": float(rng.choice([0.02, 0.05, 0.1, 0.25, 0.5, 1.0])),
    }


def run_fold(df, cols, p, spw_base, fold):
    (a0, a1), (b0, b1) = FOLDS[fold]
    # ISO date strings compare correctly as text; each fold trains strictly on earlier days than it validates on.
    d = df["event_date"].astype(str)
    tr, va = df[(d >= a0) & (d <= a1)], df[(d >= b0) & (d <= b1)]
    q = {k: v for k, v in p.items() if k != "spw_factor"}
    clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=MC["seed"], n_jobs=N_JOBS,
                            scale_pos_weight=spw_base * p["spw_factor"], **q)
    t0 = time.time()
    clf.fit(tr[cols], tr["label"])
    s = clf.predict_proba(va[cols])[:, 1]
    return float(average_precision_score(va["label"], s)), time.time() - t0, int(va["label"].sum())


def main(sets):
    fa = read_json_hdfs(f"{ROOT}/analytics/features/_ACTIVE.json")
    feat_dir = fa["path"].replace(CFG["hdfs"]["namenode_rpc"], "")
    meta = read_json_hdfs(f"{feat_dir}/_FEATURES.json")
    # Only the train and val splits are loaded; the test periods are never read by the tuning scripts.
    train = fetch_split(feat_dir, "train")
    val = fetch_split(feat_dir, "val")
    df = pd.concat([train, val], ignore_index=True)
    cats = meta["categorical"]
    # Levels and spw_base come from the full training split, so fold B (which trains on days 1 to 4) technically
    # sees category codes and a class ratio that include days 5 to 6. Minor, but not strictly time-pure.
    levels = {c: sorted(train[c].dropna().unique().tolist()) for c in cats}
    for c in cats:
        df[c] = pd.Categorical(df[c], categories=levels[c])
    os.makedirs(OUT, exist_ok=True)
    y_tr = train["label"].to_numpy()
    spw_base = (len(y_tr) - y_tr.sum()) / y_tr.sum()   # same base as stage train_xgb; the factor scales it

    for name in sets:
        spec = MC["xgboost"][name]
        cols = meta["feature_sets"][spec["feature_set"]]
        # The rng is re-seeded per model, so X1 and X2 are compared on the same candidate configurations.
        # The library defaults are scored first as the "before tuning" reference.
        rng = np.random.default_rng(MC["seed"])
        trials, best = [], None
        default = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0,
                   "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0, "spw_factor": 1.0}
        candidates = [("default", default)] + [(f"t{i:02d}", sample(rng)) for i in range(N_TRIALS)]
        for tid, p in candidates:
            scores, secs = {}, 0.0
            for fold in FOLDS:
                pr, s, npos = run_fold(df, cols, p, spw_base, fold)
                scores[fold] = round(pr, 5)
                secs += s
            mean = float(np.mean(list(scores.values())))
            trials.append({"trial": tid, "params": p, "pr_auc_fold_A": scores["A"], "pr_auc_fold_B": scores["B"],
                           "pr_auc_mean": round(mean, 5), "fit_seconds_both_folds": round(secs, 1)})
            if best is None or mean > best["pr_auc_mean"]:
                best = trials[-1]
            L.info("%s %s mean %.4f (A %.4f, B %.4f) %.0fs | best %s %.4f", name, tid, mean, scores["A"], scores["B"],
                   secs, best["trial"], best["pr_auc_mean"])
            # rewritten after every trial so partial results survive if the long run is interrupted
            with open(f"{OUT}/xgb_{name}.json", "w", encoding="utf-8") as f:
                json.dump({"model": name, "feature_set": spec["feature_set"], "n_trials": N_TRIALS, "seed": MC["seed"],
                           "selection": "mean PR-AUC of fold A and fold B", "folds": FOLDS, "spw_base": spw_base,
                           "updated_at": utcnow(), "best": best, "trials": trials}, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["X1", "X2"])
