"""Stage 4a: data mart publication.
Spark writes tables to the PostgreSQL staging schema over JDBC. After validation, one database
transaction moves them into the serving schemas (mart = analyst view, restricted = labels/metrics),
so the dashboard never sees a half-written version. mart.reviews is never replaced."""
import json
import os

import psycopg2
from pyspark.sql import functions as F

from common import CFG, get_active, hdfs_read_text, hpath, log, spark_session, utcnow, write_audit

L = log("publish_mart")
PW = os.environ["AML_DB_PASSWORD"]
JDBC = CFG["mart"]["jdbc_url"] + "?reWriteBatchedInserts=true"
PROPS = {"user": CFG["mart"]["user"], "password": PW, "driver": "org.postgresql.Driver", "batchsize": "20000"}
MODELS_SCORED = list(CFG["models"]["mllib"])
if CFG["models"].get("include_xgboost", True):
    MODELS_SCORED += [k for k, v in CFG["models"]["xgboost"].items() if isinstance(v, dict)]
ALL_MODELS = ["M0"] + MODELS_SCORED

TABLES = {  # table -> (target schema, primary key, extra indexes)
    "fact_transaction": ("mart", "transaction_id", ["event_date", "from_account_key", "to_account_key"]),
    "dim_account": ("mart", "account_key", ["bank_id"]),
    "dim_date": ("mart", "date_key", []),
    "dim_model_version": ("mart", "model_version", []),
    "predictions": ("mart", "transaction_id, model_version", ["model_version, period, risk_score DESC"]),
    "agg_daily": ("mart", "event_date, payment_format, payment_currency", []),
    "eval_labels": ("restricted", "transaction_id", []),
    "eval_metrics": ("restricted", "model, split", []),
    "pipeline_runs": ("mart", "stage", []),
}


def write(df, table, parts=4):
    # Spark recreates the staging table on every run, so it has no keys or indexes yet; they are added after the
    # bulk load (faster), and a failing primary key also acts as a uniqueness check on the data.
    df.repartition(parts).write.mode("overwrite").option("truncate", "false").jdbc(JDBC, f"staging.{table}", properties=PROPS)


def main():
    spark = spark_session("aml-s07-publish-mart")
    cur = get_active(spark, "curated")
    fa = get_active(spark, "analytics/features")
    path, batch_id = cur["path"], cur["batch_id"]
    split_of = spark.read.parquet(fa["path"]).select("transaction_id", "split")

    # Inner join with the feature table, so fact_transaction.period always agrees with the split used for modelling.
    tx = spark.read.parquet(f"{path}/transactions").join(split_of, "transaction_id").withColumnRenamed("split", "period")
    write(tx.drop("source_row"), "fact_transaction", 8)
    acc = spark.read.parquet(f"{path}/accounts")
    write(acc, "dim_account", 2)
    write(tx.select("event_date").distinct().withColumn("date_key", F.date_format("event_date", "yyyyMMdd").cast("int"))
          .join(tx.select("event_date", "period").distinct(), "event_date"), "dim_date", 1)
    agg = (tx.groupBy("event_date", "payment_format", "payment_currency")
             .agg(F.count("*").alias("tx_count"), F.sum("amount_paid").alias("amount_paid_sum")))
    write(agg, "agg_daily", 1)
    write(spark.read.parquet(f"{path}/labels").select("transaction_id", "is_laundering"), "eval_labels", 4)

    scores = spark.read.parquet(hpath("analytics", "scores")).filter(F.col("model").isin(MODELS_SCORED))
    scores = scores.select("transaction_id", "model_version", "feature_version", "source_version", "scored_at",
                           "threshold_version", "risk_score", F.col("predicted_label").cast("int"), "period", "model")
    write(scores, "predictions", 8)

    # M0 is a reference only (no predictions, no stored threshold); it gets the placeholder threshold 0.5 and appears
    # in dim_model_version / eval_metrics so the comparison tables can show it.
    mv_rows, metric_rows = [], []
    for m in ALL_MODELS:
        r = json.loads(hdfs_read_text(spark, hpath("analytics", "metrics", f"{m}.json")))
        mv = f"{m}@{r['run_id'][:8]}"
        mv_rows.append((mv, m, r["algorithm"], "xgboost" if m.startswith("X") else ("baseline" if m == "M0" else "spark-mllib"),
                        r["feature_set"], fa["feature_version"], fa["source_version"], float(r.get("threshold") or 0.5),
                        r["run_id"], r.get("registered_model"), str(r.get("registered_version")), r.get("written_at"),
                        r.get("fit_seconds"), r.get("peak_rss_mb")))
        for s in ["val", "test_main", "test_add"]:
            metric_rows.append((m, s, mv, json.dumps(r[s])))
    mv_df = spark.createDataFrame(mv_rows, "model_version string, model string, algorithm string, library string, "
                                           "feature_set string, feature_version string, source_version string, threshold double, "
                                           "run_id string, registered_model string, registered_version string, trained_at string, "
                                           "fit_seconds double, peak_rss_mb double")
    write(mv_df, "dim_model_version", 1)
    write(spark.createDataFrame(metric_rows, "model string, split string, model_version string, metrics_json string"), "eval_metrics", 1)

    # audit summary of upstream stages for the pipeline page of the dashboard
    runs = []
    for stage in ["etl", "features"]:
        txt = hdfs_read_text(spark, hpath("audit", batch_id, f"{stage}.json"))
        runs.append((stage, txt))
    write(spark.createDataFrame(runs, "stage string, audit_json string"), "pipeline_runs", 1)

    # Row counts as Spark sees them, to compare with what actually arrived in PostgreSQL (detects a lossy JDBC write).
    n_tx = tx.count()
    expected_pred = {r["model_version"]: r["n"] for r in scores.groupBy("model_version").agg(F.count("*").alias("n")).collect()}
    spark.stop()

    # ---------------- validate staging, then publish in one transaction
    conn = psycopg2.connect(host="postgres", dbname="aml", user="aml", password=PW)
    try:
        with conn:
            with conn.cursor() as c:
                c.execute("SELECT count(*) FROM staging.fact_transaction")
                n_fact = c.fetchone()[0]
                c.execute("SELECT model_version, count(*) FROM staging.predictions GROUP BY 1")
                got_pred = dict(c.fetchall())
                c.execute("SELECT count(*) FROM staging.predictions p LEFT JOIN staging.fact_transaction f "
                          "USING (transaction_id) WHERE f.transaction_id IS NULL")
                orphans = c.fetchone()[0]
                checks = {"fact_rows_equal_curated": n_fact == n_tx, "predictions_complete": got_pred == expected_pred,
                          "no_orphan_predictions": orphans == 0}
                if not all(checks.values()):
                    raise SystemExit(f"mart validation failed, active version kept: {checks}")
                for t, (schema, pk, idx) in TABLES.items():
                    c.execute(f"ALTER TABLE staging.{t} ADD PRIMARY KEY ({pk})")
                    for i, cols in enumerate(idx):
                        c.execute(f"CREATE INDEX ON staging.{t} ({cols})")
                # PostgreSQL DDL is transactional, so readers see either the old tables or the new ones, never a mix.
                # CASCADE also drops dependent views (v_alert_queue), which is recreated below. mart.reviews is not in
                # TABLES and is never touched. Labels and metrics go to the restricted schema, which analyst cannot read.
                for t, (schema, pk, idx) in TABLES.items():
                    c.execute(f"DROP TABLE IF EXISTS {schema}.{t} CASCADE")
                    c.execute(f"ALTER TABLE staging.{t} SET SCHEMA {schema}")
                    role = "analyst, researcher" if schema == "mart" else "researcher"
                    c.execute(f"GRANT SELECT ON {schema}.{t} TO {role}")
                c.execute("""CREATE OR REPLACE VIEW mart.v_alert_queue AS
                             SELECT p.model_version, p.risk_score, p.predicted_label, f.*
                             FROM mart.predictions p JOIN mart.fact_transaction f USING (transaction_id)""")
                c.execute("GRANT SELECT ON mart.v_alert_queue TO analyst, researcher")
        L.info("data mart published: %d transactions, predictions %s", n_fact, got_pred)
    finally:
        conn.close()
    os.makedirs("/opt/aml/logs/audit", exist_ok=True)
    with open(f"/opt/aml/logs/audit/{batch_id}_publish_mart.json", "w", encoding="utf-8") as f:
        json.dump({"published_at": utcnow(), "fact_rows": n_fact, "predictions": got_pred, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
