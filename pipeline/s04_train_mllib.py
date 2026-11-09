"""Stage 2b: model development WITH Spark MLlib: M0 baseline, M1 logistic regression,
M2 logistic regression + graph features, M3 random forest, M4 random forest + graph features. Tracking and registry in MLflow.

Pipeline: StringIndexer, OneHotEncoder, VectorAssembler, numeric transformation (StandardScaler),
estimator. Fitted on training data only. Class weights via weightCol from the training distribution.
Threshold chosen on validation with the rule in config.yaml and then locked for the test periods.
"""
import json
import os
import sys
import time

import mlflow
import mlflow.spark
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import CFG, get_active, hdfs_read_text, hdfs_write_text, hpath, log, spark_session, utcnow
from metrics import evaluate, threshold_max_f1

L = log("train_mllib")
MC = CFG["models"]
K = MC["precision_at_k"]
EXPERIMENT = "aml-amlworld-hi-small"
EVAL_SPLITS = ["val", "test_main", "test_add"]


def build_pipeline(cat, num, algorithm, params, seed):
    idx = [f"{c}_idx" for c in cat]
    ohe = [f"{c}_ohe" for c in cat]
    # handleInvalid="keep": a category unseen in training (possible in later periods) gets its own bucket instead of
    # crashing the job. StandardScaler has withMean=False so the one-hot vectors stay sparse (centering would densify them).
    stages = [
        StringIndexer(inputCols=cat, outputCols=idx, handleInvalid="keep"),
        OneHotEncoder(inputCols=idx, outputCols=ohe, handleInvalid="keep"),
        VectorAssembler(inputCols=ohe + num, outputCol="features_raw", handleInvalid="error"),
        StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True),
    ]
    if algorithm == "logistic_regression":
        est = LogisticRegression(featuresCol="features", labelCol="label", weightCol="w", **params)
    elif algorithm == "random_forest":
        est = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="w", seed=seed, **params)
    else:
        raise ValueError(algorithm)
    return Pipeline(stages=stages + [est])


def scores_np(model, df):
    # Collects the whole split to the driver (fine at this data size) so the metric code can be shared with XGBoost.
    pdf = (model.transform(df)
           .select("label", vector_to_array("probability").getItem(1).alias("s"))
           .toPandas())
    return pdf["label"].to_numpy(), pdf["s"].to_numpy()


def main(only=None):
    spark = spark_session("aml-s04-train-mllib", {"spark.sql.execution.arrow.pyspark.enabled": "true"})
    fa = get_active(spark, "analytics/features")
    meta = json.loads(hdfs_read_text(spark, f"{fa['path']}/_FEATURES.json"))
    feats = spark.read.parquet(fa["path"])
    seed = MC["seed"]

    # fixed row layout (hash partitioning on transaction_id + sort) so seeded estimators are reproducible
    train = (feats.filter("split = 'train'").repartition(8, "transaction_id").sortWithinPartitions("transaction_id"))
    n = train.count()
    n_pos = train.filter("label = 1").count()
    # Balanced class weights from the training split only; positive_weight_factor in config.yaml (tuned: 0.02)
    # scales the positive weight down per model, so the final weight is deliberately far below "balanced".
    w_pos, w_neg = n / (2.0 * n_pos), n / (2.0 * (n - n_pos))
    train = train.cache()
    train.count()
    evals = {s: feats.filter(f"split = '{s}'").repartition(8, "transaction_id").sortWithinPartitions("transaction_id").cache()
             for s in EVAL_SPLITS}

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(EXPERIMENT)
    common_params = {"feature_version": fa["feature_version"], "source_version": fa["source_version"],
                     "seed": seed, "class_weight_pos": round(w_pos, 4), "class_weight_neg": round(w_neg, 6),
                     "threshold_rule": MC["threshold_rule"], "precision_at_k": K, "library": "spark-mllib",
                     "train_rows": n, "train_positives": n_pos}
    results = {}

    # ---------------- M0: majority-class baseline (always predicts 0, constant score)
    if not only or "M0" in only:
        with mlflow.start_run(run_name="M0") as run:
            res = {"model": "M0", "algorithm": "majority_class", "feature_set": "none", "run_id": run.info.run_id}
            mlflow.log_params(dict(common_params, algorithm="majority_class", feature_set="none"))
            for s, df in evals.items():
                y = df.select("label").toPandas()["label"].to_numpy()
                m = evaluate(y, np.zeros(len(y)), threshold=0.5, k=K)
                res[s] = m
                mlflow.log_metrics({f"{s}_{k}": v for k, v in m.items() if isinstance(v, (int, float)) and v is not None})
            results["M0"] = res

    # ---------------- M1 to M4
    for name, spec in MC["mllib"].items():
        if only and name not in only:
            continue
        cols = meta["feature_sets"][spec["feature_set"]]
        cat = [c for c in cols if c in meta["categorical"]]
        num = [c for c in cols if c not in meta["categorical"]]
        pipe = build_pipeline(cat, num, spec["algorithm"], spec.get("params") or {}, seed)
        wf = float(spec.get("positive_weight_factor", 1.0))
        train_w = train.withColumn("w", F.when(F.col("label") == 1, F.lit(w_pos * wf)).otherwise(F.lit(w_neg)))
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(dict(common_params, algorithm=spec["algorithm"], feature_set=spec["feature_set"],
                                   n_features_input=len(cols), hyperparams=json.dumps(spec.get("params") or {}),
                                   positive_weight_factor=wf))
            mlflow.log_text("\n".join(cols), "features.txt")
            t0 = time.time()
            model = pipe.fit(train_w)
            fit_s = time.time() - t0
            t0 = time.time()
            y_val, s_val = scores_np(model, evals["val"])
            # Threshold comes from validation scores only and is reused unchanged for both test periods.
            thr, f1v = threshold_max_f1(y_val, s_val)
            res = {"model": name, "algorithm": spec["algorithm"], "feature_set": spec["feature_set"],
                   "run_id": run.info.run_id, "fit_seconds": round(fit_s, 1), "threshold": thr,
                   "threshold_val_f1": f1v}
            res["val"] = evaluate(y_val, s_val, thr, K)
            for s in ["test_main", "test_add"]:
                y, sc = scores_np(model, evals[s])
                res[s] = evaluate(y, sc, thr, K)
            res["predict_seconds_all_eval"] = round(time.time() - t0, 1)
            for s in EVAL_SPLITS:
                mlflow.log_metrics({f"{s}_{k}": v for k, v in res[s].items() if isinstance(v, (int, float)) and v is not None})
            mlflow.log_metrics({"fit_seconds": res["fit_seconds"], "threshold": thr})

            model_path = hpath("analytics", "models", name, run.info.run_id)
            model.write().overwrite().save(model_path)
            mlflow.set_tag("model_hdfs_path", model_path)
            res["model_hdfs_path"] = model_path
            try:
                info = mlflow.spark.log_model(model, artifact_path="model", registered_model_name=f"aml_{name}",
                                              pip_requirements=["pyspark==4.1.4"])
                res["registered_model"] = f"aml_{name}"
                res["registered_version"] = getattr(info, "registered_model_version", None)
            except Exception as e:  # registry problems must not lose the trained model (it is on HDFS)
                L.warning("MLflow model logging failed for %s: %s", name, e)
                res["registry_error"] = str(e)[:500]
            mlflow.log_dict(res, "metrics.json")
            results[name] = res
            L.info("%s done: fit %.1fs, val PR-AUC %s", name, fit_s, res["val"]["pr_auc"])

    for name, res in results.items():
        res["written_at"] = utcnow()
        hdfs_write_text(spark, hpath("analytics", "metrics", f"{name}.json"), json.dumps(res, indent=2))
        os.makedirs("/opt/aml/logs/metrics", exist_ok=True)
        with open(f"/opt/aml/logs/metrics/{name}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
