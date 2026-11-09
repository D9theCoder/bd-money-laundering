"""Stage 1b: ETL with Spark SQL. raw CSV (HDFS) -> validated, normalized curated Parquet (HDFS).

Implements: positional header mapping (two "Account" columns), trimming,
timestamp/amount/label/identity validation, quarantine with source row number, bank-code normalization,
account_key, transaction_id = checksum prefix + source row, Decimal type from data profile, metadata join
with count check, tables + graph representation, quality gate before publishing the curated version.
"""
import json

from pyspark.sql import functions as F
from pyspark.sql import types as T

from common import (CFG, VARIANT, get_active, hdfs_exists, hdfs_read_text, hdfs_rename, hdfs_delete, hpath, log,
                    set_active, spark_session, utcnow, write_audit)

L = log("etl")
TX_COLS = ["ts_raw", "from_bank_raw", "from_account", "to_bank_raw", "to_account", "amount_received_raw",
           "receiving_currency", "amount_paid_raw", "payment_currency", "payment_format", "label_raw"]
DEC_RE = r"^[0-9]+(\.[0-9]+)?$"


def norm_bank(c):
    """Bank code normalized through its integer representation and stored back as string ("010" -> "10")."""
    # Non-numeric codes give NULL, which is then quarantined as bank_code_invalid. decimal(38,0) avoids int overflow.
    return F.when(c.rlike(r"^[0-9]+$"), c.cast("decimal(38,0)").cast("string"))


def main():
    spark = spark_session("aml-s02-etl")
    raw = get_active(spark, "raw")
    batch_id, prefix, raw_dir = raw["batch_id"], raw["source_version"], raw["raw_dir"]
    nn = CFG["hdfs"]["namenode_rpc"]
    manifest = json.loads(hdfs_read_text(spark, f"{nn}{raw_dir}/_MANIFEST.json"))
    files = manifest["files"]
    audit = {"source_version": prefix, "raw_dir": raw_dir}

    # ------------------------------------------------------------ transactions
    # Read as plain text and parse with an explicit positional schema: the header has two columns named "Account",
    # so header names cannot be used. zipWithIndex keeps the physical line number (source_row), which is used for
    # quarantine traceability and for transaction_id.
    lines = spark.read.text(f"{nn}{raw_dir}/{files['transactions']['file']}")
    indexed = lines.rdd.zipWithIndex().map(lambda r: (r[1], r[0][0])).toDF(["line_idx", "value"])
    header_line = indexed.filter("line_idx = 0").first()["value"]
    data = indexed.filter("line_idx > 0").withColumnRenamed("line_idx", "source_row")
    schema = ", ".join(f"{c} STRING" for c in TX_COLS)
    parsed = data.select("source_row", "value", F.from_csv("value", schema).alias("r")).select("source_row", "value", "r.*")
    for c in TX_COLS:
        parsed = parsed.withColumn(c, F.trim(F.col(c)))

    parsed = (parsed
              .withColumn("event_time", F.to_timestamp("ts_raw", "yyyy/MM/dd HH:mm"))
              .withColumn("from_bank", norm_bank(F.col("from_bank_raw")))
              .withColumn("to_bank", norm_bank(F.col("to_bank_raw"))))

    # One entry per failed rule; a row is valid only when the array is empty. The coalesce(..., False) matters:
    # a NULL from rlike()/isin() would make the negated test NULL, and when() would silently treat the row as valid.
    reasons = F.array_compact(F.array(
        F.when(F.col("event_time").isNull(), F.lit("timestamp_invalid")),
        F.when(F.col("from_bank").isNull() | F.col("to_bank").isNull(), F.lit("bank_code_invalid")),
        F.when(F.coalesce(F.length("from_account"), F.lit(0)) == 0, F.lit("from_account_missing")),
        F.when(F.coalesce(F.length("to_account"), F.lit(0)) == 0, F.lit("to_account_missing")),
        F.when(~F.coalesce(F.col("amount_received_raw").rlike(DEC_RE), F.lit(False)), F.lit("amount_received_invalid")),
        F.when(~F.coalesce(F.col("amount_paid_raw").rlike(DEC_RE), F.lit(False)), F.lit("amount_paid_invalid")),
        F.when(~F.coalesce(F.col("label_raw").isin("0", "1"), F.lit(False)), F.lit("label_invalid")),
        F.when(F.coalesce(F.length("receiving_currency"), F.lit(0)) == 0, F.lit("receiving_currency_missing")),
        F.when(F.coalesce(F.length("payment_currency"), F.lit(0)) == 0, F.lit("payment_currency_missing")),
        F.when(F.coalesce(F.length("payment_format"), F.lit(0)) == 0, F.lit("payment_format_missing")),
    ))
    parsed = parsed.withColumn("reasons", reasons).cache()
    n_source = parsed.count()

    valid = parsed.filter(F.size("reasons") == 0)
    quarantine = parsed.filter(F.size("reasons") > 0).select("source_row", "value", "reasons")

    # Decimal precision/scale from the data profile
    prof = valid.select(
        *[F.max(F.length(F.split(F.col(c), r"\.").getItem(0))).alias(f"{c}_int") for c in ["amount_received_raw", "amount_paid_raw"]],
        *[F.max(F.coalesce(F.length(F.split(F.col(c), r"\.").getItem(1)), F.lit(0))).alias(f"{c}_frac") for c in ["amount_received_raw", "amount_paid_raw"]],
    ).first().asDict()
    scale = max(prof["amount_received_raw_frac"], prof["amount_paid_raw_frac"])
    precision = min(38, max(prof["amount_received_raw_int"], prof["amount_paid_raw_int"]) + scale)
    dec = T.DecimalType(precision, scale)
    audit["decimal_profile"] = dict(prof, chosen=f"decimal({precision},{scale})")

    # transaction_id is deterministic (source version + source row), so re-running the ETL on the same file
    # reproduces the same ids; the features, scores and graph all join on it.
    tx = (valid
          .withColumn("transaction_id", F.concat(F.lit(prefix + "-"), F.lpad(F.col("source_row").cast("string"), 8, "0")))
          .withColumn("from_account_key", F.concat_ws(":", "from_bank", "from_account"))
          .withColumn("to_account_key", F.concat_ws(":", "to_bank", "to_account"))
          .withColumn("amount_received", F.col("amount_received_raw").cast(dec))
          .withColumn("amount_paid", F.col("amount_paid_raw").cast(dec))
          .withColumn("is_laundering", F.col("label_raw").cast("int"))
          .withColumn("event_date", F.to_date("event_time"))
          .select("transaction_id", "source_row", "event_time", "event_date", "from_bank", "from_account",
                  "from_account_key", "to_bank", "to_account", "to_account_key", "amount_received",
                  "receiving_currency", "amount_paid", "payment_currency", "payment_format", "is_laundering"))
    tx = tx.cache()
    n_valid = tx.count()

    # ------------------------------------------------------------ account metadata
    acc_schema = T.StructType([T.StructField(n, T.StringType()) for n in
                               ["bank_name", "bank_id_raw", "account_number", "entity_id", "entity_name"]])
    acc = (spark.read.csv(f"{nn}{raw_dir}/{files['accounts']['file']}", header=True, schema=acc_schema)
           .select(*[F.trim(F.col(c)).alias(c) for c in acc_schema.fieldNames()])
           .withColumn("bank_id", norm_bank(F.col("bank_id_raw")))
           .withColumn("account_key", F.concat_ws(":", "bank_id", "account_number"))
           .drop("bank_id_raw")).cache()
    n_acc = acc.count()
    n_acc_bad = acc.filter(F.col("bank_id").isNull() | (F.length("account_number") == 0)).count()
    n_dup_keys = acc.groupBy("account_key").count().filter("count > 1").count()   # duplicates would multiply rows in the left joins below

    # ------------------------------------------------------------ join + graph representation
    active_keys = (tx.select(F.col("from_account_key").alias("account_key"))
                   .union(tx.select(F.col("to_account_key").alias("account_key"))).distinct()).cache()
    n_active = active_keys.count()
    acc_cols = acc.select("account_key", "bank_id", "bank_name", "entity_id", "entity_name")
    joined = (tx.join(acc_cols.select(F.col("account_key").alias("from_account_key"), F.col("bank_name").alias("from_bank_name")),
                      "from_account_key", "left")
              .join(acc_cols.select(F.col("account_key").alias("to_account_key"), F.col("bank_name").alias("to_bank_name")),
                    "to_account_key", "left"))
    n_joined = joined.count()
    n_active_matched = active_keys.join(acc_cols, "account_key", "inner").count()
    n_self_loops = tx.filter(F.col("from_account_key") == F.col("to_account_key")).count()
    labels = tx.groupBy("is_laundering").count().collect()
    label_counts = {int(r["is_laundering"]): r["count"] for r in labels}
    n_tx_ids = tx.select("transaction_id").distinct().count()

    # Graph nodes are only accounts that appear in an accepted transaction; accounts_out keeps every metadata
    # account and flags the active ones. Left join: an active account without metadata stays a node with NULL bank/entity.
    nodes = active_keys.join(acc_cols, "account_key", "left")
    accounts_out = acc.join(active_keys.withColumn("is_active", F.lit(True)), "account_key", "left") \
        .withColumn("is_active", F.coalesce("is_active", F.lit(False)))
    banks = acc.select("bank_id", "bank_name").distinct()
    edges = tx.select("transaction_id", F.col("from_account_key").alias("src"), F.col("to_account_key").alias("dst"),
                      "event_time", "event_date", "amount_paid", "payment_currency", "amount_received",
                      "receiving_currency", "payment_format")

    # ------------------------------------------------------------ write staging
    # Everything is written to a staging path first; it only becomes the curated version after the quality gate.
    # The label is split into its own table so that the transaction, feature and graph tables never carry it.
    stage = hpath("curated", "_staging", batch_id)
    hdfs_delete(spark, stage)
    tx.drop("is_laundering").repartition("event_date").write.mode("overwrite").partitionBy("event_date").parquet(f"{stage}/transactions")
    tx.select("transaction_id", "event_date", "is_laundering").repartition(4).write.mode("overwrite").parquet(f"{stage}/labels")
    accounts_out.repartition(2).write.mode("overwrite").parquet(f"{stage}/accounts")
    banks.coalesce(1).write.mode("overwrite").parquet(f"{stage}/graph/banks")
    nodes.repartition(2).write.mode("overwrite").parquet(f"{stage}/graph/nodes")
    edges.repartition(8).write.mode("overwrite").parquet(f"{stage}/graph/edges")
    quarantine.write.mode("overwrite").parquet(hpath("quarantine", batch_id, "transactions"))

    # Reconcile from what was actually written to HDFS, not from the in-memory DataFrames.
    s_edges = spark.read.parquet(f"{stage}/graph/edges")
    s_nodes = spark.read.parquet(f"{stage}/graph/nodes")
    n_edges = s_edges.count()
    n_nodes = s_nodes.count()
    endpoints_missing = (s_edges.select(F.col("src").alias("k")).union(s_edges.select(F.col("dst").alias("k"))).distinct()
                         .join(s_nodes.select(F.col("account_key").alias("k")), "k", "left_anti").count())

    n_quarantine = n_source - n_valid
    expected_source = files["transactions"]["lines_including_header"] - 1
    checks = {
        "source_rows_match_manifest": n_source == expected_source,
        "accepted_plus_quarantine_equals_source": n_valid + n_quarantine == n_source,
        "join_preserves_row_count": n_joined == n_valid,
        "edges_equal_accepted_transactions": n_edges == n_valid,
        "transaction_id_unique": n_tx_ids == n_valid,
        "edge_endpoints_in_nodes": endpoints_missing == 0,
        "metadata_keys_unique": n_dup_keys == 0,
    }
    audit.update({
        "transactions_source": n_source, "transactions_accepted": n_valid, "transactions_quarantined": n_quarantine,
        "label_0": label_counts.get(0, 0), "label_1": label_counts.get(1, 0),
        "label_1_pct": round(100 * label_counts.get(1, 0) / max(n_valid, 1), 4),
        "metadata_rows": n_acc, "metadata_invalid_rows": n_acc_bad, "metadata_duplicate_keys": n_dup_keys,
        "active_account_nodes": n_nodes, "inactive_metadata_accounts": n_acc - n_active_matched,
        "active_accounts_matched_to_metadata_pct": round(100 * n_active_matched / max(n_active, 1), 4),
        "edges": n_edges, "self_loops": n_self_loops,
        "event_time_min": str(tx.agg(F.min("event_time")).first()[0]),
        "event_time_max": str(tx.agg(F.max("event_time")).first()[0]),
        "checks": checks, "header_line": header_line,
    })

    # On failure the staging data is left in place for inspection and the previously active curated version stays active.
    if not all(checks.values()):
        audit["published"] = False
        write_audit(spark, batch_id, "etl", audit)
        raise SystemExit(f"quality gate failed, previous curated version kept: {checks}")

    final = hpath("curated", batch_id)
    hdfs_rename(spark, stage, final)
    set_active(spark, "curated", {"batch_id": batch_id, "source_version": prefix, "path": final,
                                  "decimal": f"decimal({precision},{scale})", "activated_at": utcnow()})
    audit["published"] = True
    audit["curated_path"] = final
    write_audit(spark, batch_id, "etl", audit)
    L.info("curated version published: %s", json.dumps({k: audit[k] for k in audit if k != "checks"}, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
