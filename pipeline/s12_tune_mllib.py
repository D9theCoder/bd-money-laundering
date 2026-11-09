"""Hyperparameter search for the Spark MLlib models (M1 and M2 logistic regression, M3 and M4 random forest).

Same protocol as s11_tune_xgboost.py: random search with a fixed seed, library defaults scored first, and every
trial fitted on two time ordered folds (A: train 1 to 6 Sept -> validate 7 to 8 Sept; B: train 1 to 4 -> validate 5 to 6).
The selection score is the mean validation PR-AUC (average precision) of the two folds. Test data is never used.
"""
import json
import os
import sys
import time

import numpy as np
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score

from common import CFG, get_active, hdfs_read_text, log, spark_session, utcnow
from s04_train_mllib import build_pipeline, scores_np

L = log("tune_mllib")
MC = CFG["models"]
N_TRIALS = int(os.environ.get("TUNE_TRIALS", 14))
OUT = "/opt/aml/logs/tuning"
FOLDS = {"A": (("2022-09-01", "2022-09-06"), ("2022-09-07", "2022-09-08")),
         "B": (("2022-09-01", "2022-09-04"), ("2022-09-05", "2022-09-06"))}

# Spark MLlib's own defaults, scored as the "before tuning" reference; w_factor 1.0 = fully balanced class weights.
DEFAULTS = {"logistic_regression": {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0, "w_factor": 1.0},
            "random_forest": {"numTrees": 20, "maxDepth": 5, "maxBins": 32, "minInstancesPerNode": 1,
                              "featureSubsetStrategy": "auto", "subsamplingRate": 1.0, "w_factor": 1.0}}


def sample(algorithm, rng):
    wf = float(rng.choice([0.02, 0.05, 0.1, 0.25, 0.5, 1.0]))
    if algorithm == "logistic_regression":
        return {"maxIter": int(rng.choice([50, 100, 200])), "regParam": float(rng.choice([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])),
                "elasticNetParam": float(rng.choice([0.0, 0.5, 1.0])), "w_factor": wf}
    return {"numTrees": int(rng.choice([20, 40, 60])), "maxDepth": int(rng.choice([5, 8, 10, 12])),
            "maxBins": int(rng.choice([32, 64])), "minInstancesPerNode": int(rng.choice([1, 5, 20])),
            "featureSubsetStrategy": str(rng.choice(["auto", "sqrt", "log2", "onethird"])),
            "subsamplingRate": float(rng.choice([0.5, 0.8, 1.0])), "w_factor": wf}


def main(names):
    spark = spark_session("aml-s12-tune-mllib", {"spark.sql.execution.arrow.pyspark.enabled": "true"})
    fa = get_active(spark, "analytics/features")
    meta = json.loads(hdfs_read_text(spark, f"{fa['path']}/_FEATURES.json"))
    # Only train and val rows are read, so test data cannot influence the selection.
    feats = spark.read.parquet(fa["path"]).filter("split in ('train','val')")
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for fold, ((a0, a1), (b0, b1)) in FOLDS.items():
        tr = feats.filter(F.col("event_date").between(F.lit(a0).cast("date"), F.lit(a1).cast("date")))
        va = feats.filter(F.col("event_date").between(F.lit(b0).cast("date"), F.lit(b1).cast("date")))
        tr = tr.repartition(8, "transaction_id").sortWithinPartitions("transaction_id").cache()
        va = va.repartition(8, "transaction_id").sortWithinPartitions("transaction_id").cache()
        n, n_pos = tr.count(), tr.filter("label = 1").count()
        data[fold] = (tr, va, n, n_pos, va.filter("label = 1").count())
        L.info("fold %s: train %d (%d pos), val positives %d", fold, n, n_pos, data[fold][4])

    for name in names:
        spec = MC["mllib"][name]
        alg = spec["algorithm"]
        cols = meta["feature_sets"][spec["feature_set"]]
        cat = [c for c in cols if c in meta["categorical"]]
        num = [c for c in cols if c not in meta["categorical"]]
        # Re-seeded per model: M1/M2 (and M3/M4) are compared on the same candidate configurations.
        rng = np.random.default_rng(MC["seed"])
        candidates = [("default", DEFAULTS[alg])] + [(f"t{i:02d}", sample(alg, rng)) for i in range(N_TRIALS)]
        trials, best = [], None
        for tid, p in candidates:
            scores, secs = {}, 0.0
            for fold, (tr, va, n, n_pos, _) in data.items():
                # Class weights come from each fold's own training distribution. (s11 uses one ratio from the full
                # training split instead.) w_factor ends up as positive_weight_factor in config.yaml.
                wf = p["w_factor"]
                w_pos, w_neg = wf * n / (2.0 * n_pos), n / (2.0 * (n - n_pos))
                trw = tr.withColumn("w", F.when(F.col("label") == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)))
                est_params = {k: v for k, v in p.items() if k != "w_factor"}
                t0 = time.time()
                model = build_pipeline(cat, num, alg, est_params, MC["seed"]).fit(trw)
                y, s = scores_np(model, va)
                secs += time.time() - t0
                scores[fold] = round(float(average_precision_score(y, s)), 5)
            mean = float(np.mean(list(scores.values())))
            trials.append({"trial": tid, "params": p, "pr_auc_fold_A": scores["A"], "pr_auc_fold_B": scores["B"],
                           "pr_auc_mean": round(mean, 5), "fit_seconds_both_folds": round(secs, 1)})
            if best is None or mean > best["pr_auc_mean"]:
                best = trials[-1]
            L.info("%s %s mean %.4f (A %.4f, B %.4f) %.0fs | best %s %.4f", name, tid, mean, scores["A"], scores["B"],
                   secs, best["trial"], best["pr_auc_mean"])
            with open(f"{OUT}/mllib_{name}.json", "w", encoding="utf-8") as f:
                json.dump({"model": name, "algorithm": alg, "feature_set": spec["feature_set"], "n_trials": N_TRIALS,
                           "seed": MC["seed"], "selection": "mean PR-AUC of fold A and fold B", "folds": FOLDS,
                           "updated_at": utcnow(), "best": best, "trials": trials}, f, indent=2)
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1:] or ["M1", "M2", "M3", "M4"])
