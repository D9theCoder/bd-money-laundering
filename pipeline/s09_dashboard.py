"""Stage 4b: static HTML dashboard (stands in for Power BI in this run).
Reads the published data mart (PostgreSQL) and investigation subgraphs (Neo4j, Cypher) and writes one
self-contained HTML file (Chart.js and Cytoscape.js inlined) to /opt/aml/dashboard/index.html.
Labels are used only for the researcher page (aggregated metrics); alert and graph views carry no labels."""
import glob
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg2
import psycopg2.extras
from neo4j import GraphDatabase

from common import CFG, log, utcnow

L = log("dashboard")
DC = CFG["dashboard"]
OUT = "/opt/aml/dashboard/index.html"
WEB = "/opt/aml/web"
PERIOD_LABEL = {"test_main": "Main test · Sep 9–10, 2022", "test_add": "Additional test · Sep 11–18, 2022"}
ROW_SQL = ("SELECT p.transaction_id, p.risk_score, p.predicted_label, f.event_time, f.from_account_key, f.to_account_key, "
           "f.from_bank, f.to_bank, f.amount_paid, f.payment_currency, f.amount_received, f.receiving_currency, f.payment_format, "
           "(SELECT r.status FROM mart.reviews r WHERE r.transaction_id = p.transaction_id AND r.model_version = p.model_version "
           " ORDER BY r.created_at DESC, r.review_id DESC LIMIT 1) AS review_status "
           "FROM mart.predictions p JOIN mart.fact_transaction f USING (transaction_id) WHERE p.model_version=%s AND p.period=%s ")


def jdefault(o):
    if isinstance(o, Decimal):
        return float(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def q(cur, sql, args=None):
    cur.execute(sql, args or ())
    return [dict(r) for r in cur.fetchall()]


def subgraph(sess, alert):
    t = alert["event_time"]
    if t.tzinfo is not None:   # stored as timestamptz in UTC by Spark JDBC; graph times are UTC local datetimes
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    # Bounded neighbourhood: hop 0 = the alert transaction itself, hop 1 = transfers touching either endpoint,
    # hop 2 = transfers touching those neighbours. Edges are ranked by time distance to the alert, so when the
    # max_edges cap truncates the result the closest transfers are the ones kept (flag "truncated").
    w = timedelta(hours=DC["subgraph_window_hours"])
    t0, t1 = (t - w).isoformat(), (t + w).isoformat()
    lim = DC["subgraph_max_edges"]
    center = [alert["from_account_key"], alert["to_account_key"]]
    rq = ("MATCH (a:Account) WHERE a.account_key IN $keys MATCH (a)-[r:TRANSFER]-(b) "
          "WHERE r.ts >= localdatetime($t0) AND r.ts <= localdatetime($t1) AND NOT r.transaction_id IN $seen "
          "WITH DISTINCT r, abs(duration.inSeconds(r.ts, localdatetime($t)).seconds) AS gap "
          "RETURN r.transaction_id AS id, startNode(r).account_key AS s, endNode(r).account_key AS d, toString(r.ts) AS ts, "
          "r.amount_paid AS amount, r.payment_currency AS cur, r.payment_format AS fmt "
          "ORDER BY gap, id LIMIT $lim")
    edges = {}
    own = sess.run("MATCH ()-[r:TRANSFER {transaction_id: $id}]->() RETURN r.transaction_id AS id, startNode(r).account_key AS s, "
                   "endNode(r).account_key AS d, toString(r.ts) AS ts, r.amount_paid AS amount, r.payment_currency AS cur, "
                   "r.payment_format AS fmt", id=alert["transaction_id"]).data()
    for e in own:
        e["hop"] = 0
        edges[e["id"]] = e
    for e in sess.run(rq, keys=center, t0=t0, t1=t1, t=t.isoformat(), seen=list(edges), lim=lim - len(edges)).data():
        e["hop"] = 1
        edges[e["id"]] = e
    truncated = len(edges) >= lim
    if DC["subgraph_max_hops"] >= 2 and len(edges) < lim:
        nbrs = sorted({k for e in edges.values() for k in (e["s"], e["d"])} - set(center))
        if nbrs:
            rest = sess.run(rq, keys=nbrs, t0=t0, t1=t1, t=t.isoformat(), seen=list(edges), lim=lim - len(edges)).data()
            for e in rest:
                e["hop"] = 2
                edges[e["id"]] = e
            truncated = len(edges) >= lim
    nodes = sorted({k for e in edges.values() for k in (e["s"], e["d"])} | set(center))
    return {"transaction_id": alert["transaction_id"], "center": center, "window": [t0, t1],
            "window_hours": DC["subgraph_window_hours"], "max_edges": lim, "max_hops": DC["subgraph_max_hops"],
            "truncated": truncated, "nodes": nodes, "edges": list(edges.values())}


def main():
    conn = psycopg2.connect(host="postgres", dbname="aml", user="aml", password=os.environ["AML_DB_PASSWORD"])
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    models = q(cur, "SELECT * FROM mart.dim_model_version ORDER BY model")
    scored = [m for m in models if m["model"] != "M0"]
    data = {"generated_at": utcnow(), "periods": PERIOD_LABEL, "models": models, "views": {}, "subgraphs": {},
            "config": {k: DC[k] for k in DC}, "precision_at_k": CFG["models"]["precision_at_k"]}

    for m in scored:
        mv = m["model_version"]
        for p in PERIOD_LABEL:
            key = f"{mv}|{p}"
            args = (mv, p)
            kpi = q(cur, "SELECT count(*) AS scored, coalesce(sum(predicted_label),0) AS alerts, avg(risk_score) AS mean_score "
                         "FROM mart.predictions WHERE model_version=%s AND period=%s", args)[0]
            daily = q(cur, "SELECT f.event_date AS d, count(*) AS scored, sum(p.predicted_label) AS alerts FROM mart.predictions p "
                           "JOIN mart.fact_transaction f USING (transaction_id) WHERE p.model_version=%s AND p.period=%s "
                           "GROUP BY 1 ORDER BY 1", args)
            fmt = q(cur, "SELECT f.payment_format AS k, count(*) AS scored, sum(p.predicted_label) AS alerts FROM mart.predictions p "
                         "JOIN mart.fact_transaction f USING (transaction_id) WHERE p.model_version=%s AND p.period=%s "
                         "GROUP BY 1 ORDER BY 3 DESC, 1", args)
            ccy = q(cur, "SELECT f.payment_currency AS k, count(*) AS alerts, sum(f.amount_paid) AS amount FROM mart.predictions p "
                         "JOIN mart.fact_transaction f USING (transaction_id) WHERE p.model_version=%s AND p.period=%s "
                         "AND p.predicted_label=1 GROUP BY 1 ORDER BY 2 DESC, 1", args)
            # snapshot rows: the top of the ranking plus non-alerts (highest-scoring near misses and a reproducible
            # md5-ordered sample), so the table also shows what the model let through; the live API pages every row
            n_top, n_neg = DC["alert_rows_per_view"], DC["nonalert_rows_per_view"]
            parts = [("top", q(cur, ROW_SQL + "ORDER BY p.risk_score DESC, p.transaction_id LIMIT %s", (mv, p, n_top))),
                     ("near", q(cur, ROW_SQL + "AND p.predicted_label = 0 ORDER BY p.risk_score DESC, p.transaction_id LIMIT %s",
                                (mv, p, n_neg // 2))),
                     ("sample", q(cur, ROW_SQL + "AND p.predicted_label = 0 ORDER BY md5(p.transaction_id) LIMIT %s",
                                  (mv, p, n_neg - n_neg // 2)))]
            rows, seen = [], set()
            for kind, part in parts:
                for r in part:
                    if r["transaction_id"] not in seen:
                        seen.add(r["transaction_id"])
                        r["src"] = kind
                        rows.append(r)
            rows.sort(key=lambda r: (-r["risk_score"], r["transaction_id"]))
            data["views"][key] = {"kpi": kpi, "daily": daily, "by_format": fmt, "by_currency": ccy, "rows": rows}

    data["overview"] = {
        "daily": q(cur, "SELECT event_date AS d, sum(tx_count) AS n FROM mart.agg_daily GROUP BY 1 ORDER BY 1"),
        "by_format": q(cur, "SELECT payment_format AS k, sum(tx_count) AS n FROM mart.agg_daily GROUP BY 1 ORDER BY 2 DESC"),
        "by_currency": q(cur, "SELECT payment_currency AS k, sum(tx_count) AS n, sum(amount_paid_sum) AS amount "
                              "FROM mart.agg_daily GROUP BY 1 ORDER BY 2 DESC"),
    }
    data["metrics"] = q(cur, "SELECT model, split, model_version, metrics_json FROM restricted.eval_metrics ORDER BY model, split")
    data["pipeline"] = {r["stage"]: json.loads(r["audit_json"]) for r in q(cur, "SELECT stage, audit_json FROM mart.pipeline_runs")}
    # Audit files are named <batch_id>_<stage>.json and batch_id itself contains one underscore, hence split("_", 2).
    # If audit files of several batches exist, the one globbed last overwrites the others.
    for f in glob.glob("/opt/aml/logs/audit/*_neo4j.json") + glob.glob("/opt/aml/logs/audit/*_publish_mart.json"):
        data["pipeline"][os.path.basename(f).split("_", 2)[-1].replace(".json", "")] = json.load(open(f, encoding="utf-8"))

    # ---------------- investigation subgraphs for the top alerts of every view
    need = {}
    for key, v in data["views"].items():
        for a in [r for r in v["rows"] if r["src"] == "top"][:DC["subgraph_alerts_per_view"]]:
            need[a["transaction_id"]] = a
    drv = GraphDatabase.driver(CFG["neo4j"]["uri"], auth=(CFG["neo4j"]["user"], os.environ["NEO4J_PASSWORD"]))
    with drv.session() as s:
        for i, (tid, a) in enumerate(need.items()):
            data["subgraphs"][tid] = subgraph(s, a)
            if i % 100 == 0:
                L.info("subgraphs %d/%d", i, len(need))
    drv.close()
    keys = sorted({n for g in data["subgraphs"].values() for n in g["nodes"]})
    data["accounts"] = {r["account_key"]: {"name": r["entity_name"], "bank": r["bank_name"]}
                        for r in q(cur, "SELECT account_key, entity_name, bank_name FROM mart.dim_account WHERE account_key = ANY(%s)", (keys,))}
    conn.close()

    # Everything is inlined into one HTML file. "</" is escaped in the embedded JSON so that a string in the data
    # can never close the surrounding <script> tag.
    tpl = open(os.path.join(WEB, "dashboard.template.html"), encoding="utf-8").read()
    ui = os.path.join(WEB, "ui")
    html = (tpl.replace("/*__AMLCSS__*/", open(os.path.join(ui, "aml.css"), encoding="utf-8").read())
               .replace("/*__AMLGRAPH__*/", open(os.path.join(ui, "aml-graph.js"), encoding="utf-8").read())
               .replace("/*__AMLUI__*/", open(os.path.join(ui, "aml-ui.js"), encoding="utf-8").read())
               .replace("/*__CHARTJS__*/", open(os.path.join(WEB, "vendor", "chart.umd.js"), encoding="utf-8").read())
               .replace("/*__CYTOSCAPE__*/", open(os.path.join(WEB, "vendor", "cytoscape.min.js"), encoding="utf-8").read())
               .replace("/*__DATA__*/null", json.dumps(data, default=jdefault, separators=(",", ":")).replace("</", "<\\/")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    L.info("dashboard written to %s (%.1f MB, %d subgraphs)", OUT, os.path.getsize(OUT) / 1e6, len(data["subgraphs"]))


if __name__ == "__main__":
    main()
