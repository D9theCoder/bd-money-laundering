"""Stage 3: graph load into Neo4j. Uses curated nodes and edges (no evaluation
labels). Account and Bank nodes, TRANSFER relationships identified by transaction_id (one per transaction,
repeated transfers kept separate), AT_BANK from account metadata. Reconciles counts after loading."""
import json
import os
import sys
import time

from neo4j import GraphDatabase
from pyspark.sql import functions as F

from common import CFG, get_active, log, spark_session, utcnow

L = log("load_neo4j")
NC = CFG["neo4j"]
BATCH = NC["batch_size"]


def batches(df, cols, size):
    buf = []
    # small partitions so the driver holds only one slice of the data at a time
    for row in df.select(*cols).repartition(max(8, df.rdd.getNumPartitions() * 8)).toLocalIterator(prefetchPartitions=False):
        buf.append(row.asDict())
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main(force=False):
    spark = spark_session("aml-s08-load-neo4j")
    cur = get_active(spark, "curated")
    path, version = cur["path"], cur["source_version"]
    drv = GraphDatabase.driver(NC["uri"], auth=(NC["user"], os.environ["NEO4J_PASSWORD"]))
    with drv.session() as s:
        # The LoadInfo node is the load marker: it is created as 'loading' and only becomes 'complete' after the counts
        # reconcile, so an interrupted load is never mistaken for a finished one and is redone on the next run.
        info = s.run("MATCH (l:LoadInfo) RETURN l.source_version AS v, l.status AS st").single()
        if info and info["v"] == version and info["st"] == "complete" and not force:
            L.info("Neo4j already holds source version %s, skipping (use --force to reload)", version)
            return
        L.info("clearing graph")
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
        for q in ["CREATE CONSTRAINT account_key IF NOT EXISTS FOR (a:Account) REQUIRE a.account_key IS UNIQUE",
                  "CREATE CONSTRAINT bank_id IF NOT EXISTS FOR (b:Bank) REQUIRE b.bank_id IS UNIQUE"]:
            s.run(q).consume()
        s.run("CREATE (:LoadInfo {source_version: $v, status: 'loading', started_at: $t})", v=version, t=utcnow()).consume()

        t0 = time.time()
        banks = spark.read.parquet(f"{path}/graph/banks").filter(F.col("bank_id").isNotNull())
        for b in batches(banks, ["bank_id", "bank_name"], BATCH):
            s.run("UNWIND $rows AS r MERGE (b:Bank {bank_id: r.bank_id}) SET b.bank_name = r.bank_name", rows=b).consume()
        # CREATE instead of MERGE for speed; this is safe because the graph was cleared above, account keys are unique
        # (checked in s02) and the uniqueness constraint would reject a duplicate anyway.
        nodes = spark.read.parquet(f"{path}/graph/nodes")
        n_nodes = 0
        for b in batches(nodes, ["account_key", "bank_id", "entity_id"], BATCH):
            s.run("UNWIND $rows AS r CREATE (:Account {account_key: r.account_key, bank_id: r.bank_id, entity_id: r.entity_id})",
                  rows=b).consume()
            n_nodes += len(b)
        s.run("MATCH (a:Account) CALL (a) { MATCH (b:Bank {bank_id: a.bank_id}) CREATE (a)-[:AT_BANK]->(b) } "
              "IN TRANSACTIONS OF 20000 ROWS").consume()
        L.info("nodes loaded: %d (%.0fs)", n_nodes, time.time() - t0)

        # Amounts become doubles (Neo4j has no decimal type), so graph amounts are for exploration only; the exact
        # values stay in Parquet and the data mart. Timestamps are stored as zone-less localdatetime.
        edges = (spark.read.parquet(f"{path}/graph/edges")
                 .withColumn("ts", F.date_format("event_time", "yyyy-MM-dd'T'HH:mm:ss"))
                 .withColumn("amount_paid", F.col("amount_paid").cast("double"))
                 .withColumn("amount_received", F.col("amount_received").cast("double")))
        cols = ["transaction_id", "src", "dst", "ts", "amount_paid", "payment_currency", "amount_received",
                "receiving_currency", "payment_format"]
        n_edges, t1 = 0, time.time()
        # CREATE keeps repeated transfers between the same pair as separate relationships (MERGE would collapse them).
        # If an endpoint were missing, MATCH would silently skip the row; the count reconciliation below catches that.
        q = ("UNWIND $rows AS r MATCH (s:Account {account_key: r.src}) MATCH (d:Account {account_key: r.dst}) "
             "CREATE (s)-[:TRANSFER {transaction_id: r.transaction_id, ts: localdatetime(r.ts), amount_paid: r.amount_paid, "
             "payment_currency: r.payment_currency, amount_received: r.amount_received, "
             "receiving_currency: r.receiving_currency, payment_format: r.payment_format}]->(d)")
        for b in batches(edges, cols, BATCH):
            s.run(q, rows=b).consume()
            n_edges += len(b)
            if n_edges % (BATCH * 25) == 0:
                L.info("edges loaded: %d (%.0f/s)", n_edges, n_edges / (time.time() - t1))
        # Relationship indexes are created after the bulk load, which is faster than maintaining them while inserting.
        for q2 in ["CREATE INDEX transfer_tx IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.transaction_id)",
                   "CREATE INDEX transfer_ts IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.ts)"]:
            s.run(q2).consume()

        g_nodes = s.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        g_edges = s.run("MATCH ()-[t:TRANSFER]->() RETURN count(t) AS c").single()["c"]
        g_atbank = s.run("MATCH ()-[t:AT_BANK]->() RETURN count(t) AS c").single()["c"]
        expected_edges = edges.count()
        ok = g_nodes == nodes.count() and g_edges == expected_edges
        s.run("MATCH (l:LoadInfo) SET l.status = $st, l.accounts = $n, l.transfers = $e, l.at_bank = $ab, l.finished_at = $t",
              st="complete" if ok else "reconciliation_failed", n=g_nodes, e=g_edges, ab=g_atbank, t=utcnow()).consume()
    drv.close()
    res = {"source_version": version, "accounts": g_nodes, "transfers": g_edges, "at_bank": g_atbank,
           "expected_transfers": expected_edges, "reconciled": ok, "seconds": round(time.time() - t0, 1)}
    os.makedirs("/opt/aml/logs/audit", exist_ok=True)
    with open(f"/opt/aml/logs/audit/{cur['batch_id']}_neo4j.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    spark.stop()
    if not ok:
        raise SystemExit(f"Neo4j reconciliation failed: {res}")
    L.info("graph loaded: %s", res)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
