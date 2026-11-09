"""Investigation API: live queries over the PostgreSQL data mart (SQL) and Neo4j (Cypher).
Reads with the `analyst` role, which has no access to the restricted schema, so evaluation labels
can never be returned by this service. The only write is an insert into mart.reviews."""
import os
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import psycopg2.extras
import psycopg2.pool
import yaml
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j import READ_ACCESS, GraphDatabase
from pydantic import BaseModel, Field

CFG = yaml.safe_load(open("/opt/aml/pipeline/config.yaml", encoding="utf-8"))
DC = CFG["dashboard"]

app = FastAPI(title="AML investigation API", version="1.0")
pool = psycopg2.pool.ThreadedConnectionPool(
    1, 5, host="postgres", dbname="aml", user="analyst", password=os.environ["ANALYST_DB_PASSWORD"],
    cursor_factory=psycopg2.extras.RealDictCursor)
driver = GraphDatabase.driver(CFG["neo4j"]["uri"], auth=(CFG["neo4j"]["user"], os.environ["NEO4J_PASSWORD"]))

PERIODS = {"test_main", "test_add"}


def sql(query, args=(), write=False):
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, args)
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
        conn.commit() if write else conn.rollback()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def check_period(period):
    if period not in PERIODS:
        raise HTTPException(422, f"period must be one of {sorted(PERIODS)}")


@app.get("/api/health")
def health():
    sql("SELECT 1")
    with driver.session(default_access_mode=READ_ACCESS) as s:
        s.run("RETURN 1").consume()
    return {"status": "ok", "postgres": True, "neo4j": True}


@app.get("/api/models")
def models():
    return sql("SELECT model_version, model, algorithm, library, feature_set, threshold, trained_at "
               "FROM mart.dim_model_version WHERE model <> 'M0' ORDER BY model")


@app.get("/api/facets")
def facets():
    return {"payment_format": [r["k"] for r in sql("SELECT DISTINCT payment_format AS k FROM mart.agg_daily ORDER BY 1")],
            "payment_currency": [r["k"] for r in sql("SELECT DISTINCT payment_currency AS k FROM mart.agg_daily ORDER BY 1")]}


SORTS = {"score_desc": "p.risk_score DESC, p.transaction_id", "score_asc": "p.risk_score ASC, p.transaction_id",
         "amount_desc": "f.amount_paid DESC, p.transaction_id", "amount_asc": "f.amount_paid ASC, p.transaction_id",
         "time_desc": "f.event_time DESC, p.transaction_id", "time_asc": "f.event_time ASC, p.transaction_id"}
LATEST_REVIEW = ("(SELECT r.status FROM mart.reviews r WHERE r.transaction_id = p.transaction_id AND r.model_version = p.model_version "
                 "ORDER BY r.created_at DESC, r.review_id DESC LIMIT 1)")
LATEST_BY_STATUS = ("SELECT transaction_id FROM (SELECT DISTINCT ON (transaction_id) transaction_id, status FROM mart.reviews "
                    "WHERE model_version = %s ORDER BY transaction_id, created_at DESC, review_id DESC) x WHERE status = %s")


def queue_filter(model_version: str, period: str, prediction: Literal["all", "alert", "clear"] = "all",
                 min_score: Optional[float] = Query(None, ge=0, le=1), max_score: Optional[float] = Query(None, ge=0, le=1),
                 payment_format: Optional[str] = None, payment_currency: Optional[str] = None, account: Optional[str] = None,
                 min_amount: Optional[float] = Query(None, ge=0), max_amount: Optional[float] = Query(None, ge=0),
                 review_status: Optional[Literal["unreviewed", "open", "in_review", "escalated", "closed"]] = None):
    check_period(period)
    where, args = ["p.model_version = %s", "p.period = %s"], [model_version, period]
    if prediction != "all":
        where.append("p.predicted_label = %s")
        args.append(1 if prediction == "alert" else 0)
    for cond, val in (("p.risk_score >= %s", min_score), ("p.risk_score <= %s", max_score), ("f.payment_format = %s", payment_format),
                      ("f.payment_currency = %s", payment_currency), ("f.amount_paid >= %s", min_amount),
                      ("f.amount_paid <= %s", max_amount)):
        if val not in (None, ""):
            where.append(cond)
            args.append(val)
    if account:
        where.append("(f.from_account_key = %s OR f.to_account_key = %s)")
        args += [account, account]
    if review_status == "unreviewed":
        where.append("NOT EXISTS (SELECT 1 FROM mart.reviews r WHERE r.transaction_id = p.transaction_id AND r.model_version = p.model_version)")
    elif review_status:
        where.append(f"p.transaction_id IN ({LATEST_BY_STATUS})")
        args += [model_version, review_status]
    # the count query can skip the fact join when no filter touches fact columns
    return " AND ".join(where), args, any(w.startswith(("f.", "(f.")) for w in where)


@app.get("/api/alerts")
def alerts(flt=Depends(queue_filter), page: int = Query(1, ge=1, le=100000), page_size: int = Query(50, ge=1, le=200),
           sort: Literal[tuple(SORTS)] = "score_desc"):
    """One page of scored transactions (alerts and non-alerts). Fetches page_size + 1 rows to know if a next page exists;
    the total is a separate call (/api/alerts/count) because counting 0.8 M joined rows takes a few seconds."""
    where, args, _ = flt
    rows = sql("SELECT p.transaction_id, p.risk_score, p.predicted_label, f.event_time, f.from_account_key, f.to_account_key, "
               "f.amount_paid, f.payment_currency, f.payment_format, " + LATEST_REVIEW + " AS review_status "
               "FROM mart.predictions p JOIN mart.fact_transaction f USING (transaction_id) "
               "WHERE " + where + " ORDER BY " + SORTS[sort] + " LIMIT %s OFFSET %s",
               args + [page_size + 1, (page - 1) * page_size])
    return {"page": page, "page_size": page_size, "sort": sort, "has_more": len(rows) > page_size, "rows": rows[:page_size]}


@app.get("/api/alerts/count")
def alerts_count(flt=Depends(queue_filter)):
    where, args, needs_fact = flt
    r = sql("SELECT count(*) AS total, coalesce(sum(p.predicted_label), 0) AS alerts FROM mart.predictions p " +
            ("JOIN mart.fact_transaction f USING (transaction_id) " if needs_fact else "") + "WHERE " + where, args)[0]
    return {"total": r["total"], "alerts": int(r["alerts"])}


@app.get("/api/transactions/{transaction_id}")
def transaction(transaction_id: str):
    rows = sql("SELECT * FROM mart.fact_transaction WHERE transaction_id = %s", (transaction_id,))
    if not rows:
        raise HTTPException(404, "transaction not found")
    return {"transaction": rows[0],
            "predictions": sql("SELECT model_version, model, risk_score, predicted_label, scored_at FROM mart.predictions "
                               "WHERE transaction_id = %s ORDER BY model", (transaction_id,)),
            "reviews": sql("SELECT review_id, model_version, status, note, reviewer, created_at FROM mart.reviews "
                           "WHERE transaction_id = %s ORDER BY created_at DESC, review_id DESC", (transaction_id,))}


EDGE_RETURN = ("r.transaction_id AS id, startNode(r).account_key AS s, endNode(r).account_key AS d, toString(r.ts) AS ts, "
               "r.amount_paid AS amount, r.payment_currency AS cur, r.payment_format AS fmt")


def to_utc_naive(t):
    # timestamptz comes back with an offset; graph timestamps are UTC local datetimes
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def expand(sess, keys, t, window, seen, lim):
    t0, t1 = (t - window).isoformat(), (t + window).isoformat()
    return sess.run("MATCH (a:Account) WHERE a.account_key IN $keys MATCH (a)-[r:TRANSFER]-(b) "
                    "WHERE r.ts >= localdatetime($t0) AND r.ts <= localdatetime($t1) AND NOT r.transaction_id IN $seen "
                    "WITH DISTINCT r, abs(duration.inSeconds(r.ts, localdatetime($t)).seconds) AS gap "
                    f"RETURN {EDGE_RETURN} ORDER BY gap, id LIMIT $lim",
                    keys=keys, t0=t0, t1=t1, t=t.isoformat(), seen=seen, lim=lim).data()


def build_subgraph(center, t, hours, max_edges, max_hops, own_edge=None):
    window = timedelta(hours=hours)
    edges = {}
    with driver.session(default_access_mode=READ_ACCESS) as sess:
        if own_edge:
            for e in sess.run(f"MATCH ()-[r:TRANSFER {{transaction_id: $id}}]->() RETURN {EDGE_RETURN}", id=own_edge).data():
                e["hop"] = 0
                edges[e["id"]] = e
        for e in expand(sess, center, t, window, list(edges), max_edges - len(edges)):
            e["hop"] = 1
            edges[e["id"]] = e
        if max_hops >= 2 and len(edges) < max_edges:
            nbrs = sorted({k for e in edges.values() for k in (e["s"], e["d"])} - set(center))
            if nbrs:
                for e in expand(sess, nbrs, t, window, list(edges), max_edges - len(edges)):
                    e["hop"] = 2
                    edges[e["id"]] = e
    nodes = sorted({k for e in edges.values() for k in (e["s"], e["d"])} | set(center))
    accounts = {r["account_key"]: {"name": r["entity_name"], "bank": r["bank_name"]}
                for r in sql("SELECT account_key, entity_name, bank_name FROM mart.dim_account WHERE account_key = ANY(%s)", (nodes,))}
    return {"center": center, "accounts": accounts, "window": [(t - window).isoformat(), (t + window).isoformat()], "window_hours": hours,
            "max_edges": max_edges, "max_hops": max_hops, "truncated": len(edges) >= max_edges, "nodes": nodes,
            "edges": list(edges.values())}


@app.get("/api/subgraph/{transaction_id}")
def subgraph_of_transaction(transaction_id: str, hours: int = Query(DC["subgraph_window_hours"], ge=1, le=168),
                            max_edges: int = Query(DC["subgraph_max_edges"], ge=1, le=200),
                            max_hops: int = Query(DC["subgraph_max_hops"], ge=1, le=2)):
    rows = sql("SELECT event_time, from_account_key, to_account_key FROM mart.fact_transaction WHERE transaction_id = %s",
               (transaction_id,))
    if not rows:
        raise HTTPException(404, "transaction not found")
    r = rows[0]
    g = build_subgraph([r["from_account_key"], r["to_account_key"]], to_utc_naive(r["event_time"]), hours, max_edges, max_hops,
                       transaction_id)
    g["transaction_id"] = transaction_id
    return g


@app.get("/api/accounts/{account_key}/subgraph")
def subgraph_of_account(account_key: str, around: datetime, hours: int = Query(DC["subgraph_window_hours"], ge=1, le=168),
                        max_edges: int = Query(DC["subgraph_max_edges"], ge=1, le=200),
                        max_hops: int = Query(DC["subgraph_max_hops"], ge=1, le=2)):
    if not sql("SELECT 1 FROM mart.dim_account WHERE account_key = %s", (account_key,)):
        raise HTTPException(404, "account not found")
    return build_subgraph([account_key], to_utc_naive(around), hours, max_edges, max_hops)


class ReviewIn(BaseModel):
    transaction_id: str
    model_version: str
    status: Literal["open", "in_review", "escalated", "closed"]
    note: Optional[str] = Field(None, max_length=2000)
    reviewer: Optional[str] = Field(None, max_length=60)


@app.post("/api/reviews", status_code=201)
def add_review(r: ReviewIn):
    if not sql("SELECT 1 FROM mart.predictions WHERE transaction_id = %s AND model_version = %s", (r.transaction_id, r.model_version)):
        raise HTTPException(404, "no prediction for this transaction and model_version")
    return sql("INSERT INTO mart.reviews (transaction_id, model_version, status, note, reviewer) "
               "VALUES (%s, %s, %s, %s, coalesce(nullif(%s, ''), current_user)) "
               "RETURNING review_id, transaction_id, model_version, status, note, reviewer, created_at",
               (r.transaction_id, r.model_version, r.status, r.note, r.reviewer), write=True)[0]


@app.get("/api/reviews")
def list_reviews(transaction_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    where, args = ("WHERE transaction_id = %s", [transaction_id]) if transaction_id else ("", [])
    return sql(f"SELECT review_id, transaction_id, model_version, status, note, reviewer, created_at FROM mart.reviews {where} "
               "ORDER BY created_at DESC, review_id DESC LIMIT %s", args + [limit])


@app.get("/")
def index():
    return FileResponse("/opt/aml/web/investigation.html")


app.mount("/dashboard", StaticFiles(directory="/opt/aml/dashboard", html=True), name="dashboard")
app.mount("/ui", StaticFiles(directory="/opt/aml/web/ui"), name="ui")
app.mount("/vendor", StaticFiles(directory="/opt/aml/web/vendor"), name="vendor")
