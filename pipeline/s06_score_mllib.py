"""Stage 2d: batch scoring with Spark MLlib, separate from training.
Loads each trained pipeline (HDFS path recorded in MLflow and the metrics file), scores the scoring
periods and writes risk_score + version metadata. Predictions are reconciled with the scored rows
before the output is accepted."""
import json
import sys
from datetime import datetime, timezone

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import CFG, get_active, hdfs_delete, hdfs_read_text, hdfs_rename, hpath, log, spark_session

L = log("score_mllib")
MC = CFG["models"]


def main(only=None):
    spark = spark_session("aml-s06-score-mllib")
    fa = get_active(spark, "analytics/features")
    periods = CFG["scoring"]["periods"]
    # Only the scoring periods (test_main, test_add) are scored, and the label is dropped before the model sees the rows.
    feats = spark.read.parquet(fa["path"]).filter(F.col("split").isin(periods)).drop("label").cache()
    n_expected = feats.count()
    scored_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for name in MC["mllib"]:
        if only and name not in only:
            continue
        res = json.loads(hdfs_read_text(spark, hpath("analytics", "metrics", f"{name}.json")))
        model = PipelineModel.load(res["model_hdfs_path"])
        thr = res["threshold"]   # the threshold locked at training time, not recomputed here
        model_version = f"{name}@{res['run_id'][:8]}"
        out = (model.transform(feats)
               .select("transaction_id",
                       F.lit(model_version).alias("model_version"),
                       F.lit(fa["feature_version"]).alias("feature_version"),
                       F.lit(fa["source_version"]).alias("source_version"),
                       F.lit(scored_at).alias("scored_at"),
                       F.lit(f"{MC['threshold_rule']}:{thr:.6f}").alias("threshold_version"),
                       vector_to_array("probability").getItem(1).alias("risk_score"),
                       F.col("split").alias("period")))
        out = out.withColumn("predicted_label", (F.col("risk_score") >= F.lit(thr)).cast("int"))
        dest = hpath("analytics", "scores", f"model={name}")
        tmp = hpath("analytics", "_tmp", f"scores_{name}")
        # Write to a temp path and validate before touching the live scores. The delete + rename below is not atomic,
        # but a failure between them is recoverable by re-running the stage.
        out.repartition(4).write.mode("overwrite").parquet(tmp)
        chk = spark.read.parquet(tmp)
        n, n_ids = chk.count(), chk.select("transaction_id").distinct().count()
        if n != n_expected or n_ids != n_expected:
            raise SystemExit(f"{name}: scored {n} rows / {n_ids} ids, expected {n_expected}")
        hdfs_delete(spark, dest)
        hdfs_rename(spark, tmp, dest)
        L.info("%s scored %d transactions (%s)", name, n, model_version)
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
