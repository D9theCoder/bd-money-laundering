"""Stage 2c: model development WITHOUT Spark MLlib (XGBoost on one machine).

The feature table produced by Spark is copied from HDFS (WebHDFS) to the local disk of one container and
loaded into pandas. Same features, same time split, same class-weight principle (scale_pos_weight from the
training distribution), same threshold rule and the same metric code as the MLlib models.
Also scores the test periods, because this path has no cluster.
"""
import json
import os
import resource
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

import webhdfs
from common import CFG, ROOT, log, utcnow
from metrics import evaluate, threshold_max_f1

L = log("train_xgboost")
MC = CFG["models"]
XC = MC["xgboost"]
K = MC["precision_at_k"]
LOCAL = "/tmp/aml_xgb"
EXPERIMENT = "aml-amlworld-hi-small"


def read_json_hdfs(path):
    return json.loads(b"".join(webhdfs.iter_file(path)).decode("utf-8"))


def fetch_split(feat_dir, split):
    """Copy one split partition (Parquet files) from HDFS to local disk and load it with pandas."""
    hdir = f"{feat_dir}/split={split}"
    ldir = f"{LOCAL}/split={split}"
    os.makedirs(ldir, exist_ok=True)
    for st in webhdfs.list_dir(hdir):
        if st["pathSuffix"].endswith(".parquet"):
            webhdfs.download(f"{hdir}/{st['pathSuffix']}", f"{ldir}/{st['pathSuffix']}")
    # fixed row order so training is reproducible regardless of the Parquet file layout
    return pd.read_parquet(ldir).sort_values("transaction_id", kind="stable").reset_index(drop=True)


def peak_rss_mb():
    # ru_maxrss is in KiB on Linux (the container); it would be bytes on macOS
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main(only=None):
    t_all = time.time()
    shutil.rmtree(LOCAL, ignore_errors=True)
    fa = read_json_hdfs(f"{ROOT}/analytics/features/_ACTIVE.json")
    feat_dir = fa["path"].replace(CFG["hdfs"]["namenode_rpc"], "")
    meta = read_json_hdfs(f"{feat_dir}/_FEATURES.json")

    t0 = time.time()
    data = {s: fetch_split(feat_dir, s) for s in ["train", "val", "test_main", "test_add"]}
    load_s = time.time() - t0
    rss_after_load = peak_rss_mb()
    cats = meta["categorical"]
    # Category levels are fixed from the training split so every split gets the same encoding.
    levels = {c: sorted(data["train"][c].dropna().unique().tolist()) for c in cats}
    for df in data.values():
        for c in cats:
            df[c] = pd.Categorical(df[c], categories=levels[c])   # unseen categories become missing

    y_tr = data["train"]["label"].to_numpy()
    n, n_pos = len(y_tr), int(y_tr.sum())
    spw = (n - n_pos) / n_pos   # fully balanced; the tuned value in config.yaml overrides it (see params below)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(EXPERIMENT)
    scored_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for name, spec in XC.items():
        # XC also holds plain settings such as n_jobs, so only dict entries are models
        if not isinstance(spec, dict) or (only and name not in only):
            continue
        cols = meta["feature_sets"][spec["feature_set"]]
        # spec params are applied last on purpose: the tuned scale_pos_weight (0.02 x N_neg/N_pos) replaces the balanced default
        params = {"tree_method": "hist", "enable_categorical": True, "scale_pos_weight": spw,
                  "random_state": MC["seed"], "n_jobs": XC["n_jobs"], **(spec.get("params") or {})}
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"library": "xgboost", "xgboost_version": xgb.__version__, "algorithm": "xgboost",
                               "feature_set": spec["feature_set"], "n_features_input": len(cols),
                               "feature_version": fa["feature_version"], "source_version": fa["source_version"],
                               "seed": MC["seed"], "scale_pos_weight": round(params["scale_pos_weight"], 4),
                               "scale_pos_weight_balanced": round(spw, 4), "n_jobs": XC["n_jobs"],
                               "threshold_rule": MC["threshold_rule"], "precision_at_k": K,
                               "train_rows": n, "train_positives": n_pos, "hyperparams": json.dumps(spec.get("params") or {})})
            clf = xgb.XGBClassifier(**params)
            t0 = time.time()
            clf.fit(data["train"][cols], y_tr)
            fit_s = time.time() - t0
            res = {"model": name, "algorithm": "xgboost", "feature_set": spec["feature_set"], "run_id": run.info.run_id,
                   "fit_seconds": round(fit_s, 1), "hdfs_to_pandas_seconds": round(load_s, 1),
                   "peak_rss_mb_after_load": rss_after_load}
            # Same rule as the MLlib path: threshold from validation only, then locked for the test periods.
            sv = clf.predict_proba(data["val"][cols])[:, 1]
            thr, f1v = threshold_max_f1(data["val"]["label"].to_numpy(), sv)
            res.update(threshold=thr, threshold_val_f1=f1v)
            res["val"] = evaluate(data["val"]["label"].to_numpy(), sv, thr, K)
            model_version = f"{name}@{run.info.run_id[:8]}"
            for s in ["test_main", "test_add"]:
                sc = clf.predict_proba(data[s][cols])[:, 1]
                res[s] = evaluate(data[s]["label"].to_numpy(), sc, thr, K)
                out = pd.DataFrame({
                    "transaction_id": data[s]["transaction_id"], "model_version": model_version,
                    "feature_version": fa["feature_version"], "source_version": fa["source_version"],
                    "scored_at": scored_at, "threshold_version": f"{MC['threshold_rule']}:{thr:.6f}",
                    "risk_score": sc, "predicted_label": (sc >= thr).astype("int32"), "period": s})
                lp = f"{LOCAL}/scores_{name}_{s}.parquet"
                out.to_parquet(lp, index=False)
                webhdfs.mkdirs(f"{ROOT}/analytics/scores/model={name}")
                webhdfs.put_file(lp, f"{ROOT}/analytics/scores/model={name}/{s}.parquet")
            res["peak_rss_mb"] = peak_rss_mb()
            res["model_version"] = model_version
            for s in ["val", "test_main", "test_add"]:
                mlflow.log_metrics({f"{s}_{k}": v for k, v in res[s].items() if isinstance(v, (int, float)) and v is not None})
            mlflow.log_metrics({"fit_seconds": res["fit_seconds"], "threshold": thr, "peak_rss_mb": res["peak_rss_mb"]})
            lm = f"{LOCAL}/{name}_model.json"
            clf.save_model(lm)
            webhdfs.mkdirs(f"{ROOT}/analytics/models/{name}/{run.info.run_id}")
            webhdfs.put_file(lm, f"{ROOT}/analytics/models/{name}/{run.info.run_id}/model.json")
            res["model_hdfs_path"] = f"{ROOT}/analytics/models/{name}/{run.info.run_id}/model.json"
            try:
                info = mlflow.xgboost.log_model(clf, name="model", registered_model_name=f"aml_{name}")
                res["registered_model"] = f"aml_{name}"
                res["registered_version"] = getattr(info, "registered_model_version", None)
            except Exception as e:
                L.warning("MLflow model logging failed for %s: %s", name, e)
                res["registry_error"] = str(e)[:500]
            res["written_at"] = utcnow()
            mlflow.log_dict(res, "metrics.json")
            lj = f"{LOCAL}/{name}.json"
            with open(lj, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            webhdfs.put_file(lj, f"{ROOT}/analytics/metrics/{name}.json")
            os.makedirs("/opt/aml/logs/metrics", exist_ok=True)
            shutil.copy(lj, f"/opt/aml/logs/metrics/{name}.json")
            L.info("%s done: fit %.1fs, val PR-AUC %s, peak RSS %s MB", name, fit_s, res["val"]["pr_auc"], res["peak_rss_mb"])
    shutil.rmtree(LOCAL, ignore_errors=True)
    L.info("xgboost stage finished in %.1fs", time.time() - t_all)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
