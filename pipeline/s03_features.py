"""Stage 2a: feature construction with PySpark / Spark SQL (feature_version v1).

History rule: for a transaction at time t and window w, history is t - w <= event_time < t and
events in the same minute as t are excluded (timestamps have minute resolution), implemented as a range
frame [t - w, t - 60 s]. Amount aggregates are computed per currency. IDs, names, labels and simulator
pattern annotations are not used as predictors.

Interpretation notes:
- frequency and total amount incoming or outgoing: sender outgoing and receiver incoming activity.
- graph features use the full history before t (no window).
"""
import json

from pyspark.sql import Window
from pyspark.sql import functions as F

from common import CFG, date_ranges, get_active, hdfs_write_text, hpath, log, set_active, spark_session, utcnow, write_audit

L = log("features")
FV = CFG["features"]["version"]
LAG = 60 if CFG["features"]["same_minute_excluded"] else 0


def unique_counterparties_prior(tx, acct_col, other_col, name):
    """Distinct counterparties of acct seen strictly before the minute of each transaction.
    A counterparty counts from the time of its first transfer with this account."""
    # Window functions cannot do a distinct count, so: one "+1" event per (account, counterparty) at their first
    # contact, merged with the transaction rows (+0). A running sum up to t - LAG then equals the distinct
    # counterparties seen before t. Rows with transaction_id NULL are only helper events and are dropped at the end.
    first = tx.groupBy(acct_col, other_col).agg(F.min("ts").alias("t"))
    ev = (first.select(F.col(acct_col).alias("acct"), "t", F.lit(1).alias("inc"), F.lit(None).cast("string").alias("transaction_id"))
          .unionByName(tx.select(F.col(acct_col).alias("acct"), F.col("ts").alias("t"), F.lit(0).alias("inc"), "transaction_id")))
    w = Window.partitionBy("acct").orderBy("t").rangeBetween(Window.unboundedPreceding, -LAG)
    return (ev.withColumn(name, F.coalesce(F.sum("inc").over(w), F.lit(0)))
              .filter(F.col("transaction_id").isNotNull()).select("transaction_id", name))


def main():
    spark = spark_session("aml-s03-features")
    cur = get_active(spark, "curated")
    batch_id, prefix, path = cur["batch_id"], cur["source_version"], cur["path"]
    tx = spark.read.parquet(f"{path}/transactions")
    labels = spark.read.parquet(f"{path}/labels").select("transaction_id", "is_laundering")

    tx = (tx.withColumn("ts", F.unix_timestamp("event_time"))
            .withColumn("amount_paid_d", F.col("amount_paid").cast("double"))
            .withColumn("amount_received_d", F.col("amount_received").cast("double")))

    # ---------------- transaction features
    f = (tx.withColumn("log_amount_paid", F.log1p("amount_paid_d"))
           .withColumn("hour", F.hour("event_time"))
           .withColumn("diff_bank", (F.col("from_bank") != F.col("to_bank")).cast("int")))

    # ---------------- non-graph history features (6 h and 24 h)
    # rangeBetween works on the value of ts (seconds), not on row counts. The upper bound -LAG keeps the current
    # minute (and so the transaction itself) out of its own history. Counts ignore currency; amount sums are
    # per currency because amounts in different currencies cannot be added. Empty window: count = 0, sum -> 0.0.
    base_num = ["amount_paid_d", "log_amount_paid", "hour", "diff_bank"]
    for h in CFG["features"]["windows_hours"]:
        W = h * 3600
        w_snd = Window.partitionBy("from_account_key").orderBy("ts").rangeBetween(-W, -LAG)
        w_snd_cur = Window.partitionBy("from_account_key", "payment_currency").orderBy("ts").rangeBetween(-W, -LAG)
        w_rcv = Window.partitionBy("to_account_key").orderBy("ts").rangeBetween(-W, -LAG)
        w_rcv_cur = Window.partitionBy("to_account_key", "receiving_currency").orderBy("ts").rangeBetween(-W, -LAG)
        f = (f.withColumn(f"snd_out_cnt_{h}h", F.count(F.lit(1)).over(w_snd))
              .withColumn(f"snd_out_amt_{h}h", F.coalesce(F.sum("amount_paid_d").over(w_snd_cur), F.lit(0.0)))
              .withColumn(f"rcv_in_cnt_{h}h", F.count(F.lit(1)).over(w_rcv))
              .withColumn(f"rcv_in_amt_{h}h", F.coalesce(F.sum("amount_received_d").over(w_rcv_cur), F.lit(0.0))))
        base_num += [f"snd_out_cnt_{h}h", f"snd_out_amt_{h}h", f"rcv_in_cnt_{h}h", f"rcv_in_amt_{h}h"]

    # ---------------- graph features (history before t)
    w_pair = Window.partitionBy("from_account_key", "to_account_key").orderBy("ts").rangeBetween(Window.unboundedPreceding, -LAG)
    f = f.withColumn("pair_prior_cnt", F.count(F.lit(1)).over(w_pair))
    snd_u = unique_counterparties_prior(tx, "from_account_key", "to_account_key", "snd_uniq_rcv_prior")
    rcv_u = unique_counterparties_prior(tx, "to_account_key", "from_account_key", "rcv_uniq_snd_prior")
    f = f.join(snd_u, "transaction_id").join(rcv_u, "transaction_id")
    graph_num = ["pair_prior_cnt", "snd_uniq_rcv_prior", "rcv_uniq_snd_prior"]
    cat = ["payment_format", "payment_currency", "receiving_currency"]

    # Split by event date from config. A date outside every range gets split = NULL, which fails the row check below.
    split = F.lit(None).cast("string")
    for name, (d0, d1) in date_ranges().items():
        split = F.when(F.col("event_date").between(F.lit(d0).cast("date"), F.lit(d1).cast("date")), F.lit(name)).otherwise(split)
    # The label is attached only as the target column "label"; it is not part of any feature list below.
    out = (f.join(labels, "transaction_id")
            .withColumn("split", split)
            .select("transaction_id", "event_time", "event_date", "from_account_key", "to_account_key", "split",
                    *cat, *base_num, *graph_num, F.col("is_laundering").alias("label")))

    dest = hpath("analytics", "features", f"feature_version={FV}", f"source={prefix}")
    out.repartition("split").write.mode("overwrite").partitionBy("split").parquet(dest)

    feats = spark.read.parquet(dest)
    counts = {r["split"]: {"rows": r["rows"], "positives": r["pos"]} for r in
              feats.groupBy("split").agg(F.count("*").alias("rows"), F.sum("label").alias("pos")).collect()}
    n_cur = spark.read.parquet(f"{path}/transactions").count()
    n_feat = sum(v["rows"] for v in counts.values())
    meta = {"feature_version": FV, "source_version": prefix, "batch_id": batch_id, "path": dest,
            "categorical": cat, "base_numeric": base_num, "graph_numeric": graph_num,
            "feature_sets": {"base": cat + base_num, "base_graph": cat + base_num + graph_num},
            "windows_hours": CFG["features"]["windows_hours"], "same_minute_lag_seconds": LAG,
            "split_counts": counts, "rows": n_feat, "curated_rows": n_cur, "created_at": utcnow()}
    # The joins above are inner joins, so a lost or duplicated row shows up as a count mismatch here.
    if n_feat != n_cur or None in counts:
        write_audit(spark, batch_id, "features", dict(meta, published=False))
        raise SystemExit("feature row count does not match curated transactions or rows outside split ranges")
    hdfs_write_text(spark, f"{dest}/_FEATURES.json", json.dumps(meta, indent=2))
    set_active(spark, "analytics/features", {"feature_version": FV, "source_version": prefix, "batch_id": batch_id,
                                              "path": dest, "activated_at": utcnow()})
    write_audit(spark, batch_id, "features", dict(meta, published=True))
    L.info("features written: %s", json.dumps(counts))
    spark.stop()


if __name__ == "__main__":
    main()
