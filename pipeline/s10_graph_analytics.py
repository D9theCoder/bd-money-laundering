"""Stage 3b: graph analytics on the Neo4j graph with the Graph Data Science library.
Algorithms: degree, PageRank, weakly connected components (WCC) and Louvain community detection, run on an
in-memory projection of the whole period (account pairs, self-loops excluded, weight = number of transfers).
The graph holds no labels. After the algorithms have run, the evaluation labels are read from the restricted
schema and used only to summarise how laundering-labelled transfers sit in the graph (researcher analysis,
aggregates only). Results: node properties in Neo4j, JSON and CSV files in /opt/aml/logs/graph."""
import json
import os
import time

import numpy as np
import pandas as pd
import psycopg2
from neo4j import GraphDatabase

from common import CFG, log, utcnow

L = log("graph_analytics")
NC = CFG["neo4j"]
GC = CFG["graph_analytics"]
OUT = "/opt/aml/logs/graph"
G = "aml_pairs"
PROPS = ["out_pairs", "in_pairs", "out_tx", "in_tx", "pagerank", "wcc", "community"]


def run(s, q, **p):
    return s.run(q, **p).data()


def timed(label, fn):
    t = time.time()
    r = fn()
    sec = round(time.time() - t, 1)
    L.info("%s: %.1fs", label, sec)
    return r, sec


def dist_summary(series):
    s = pd.Series(series).dropna().astype(float)
    return {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
            "p90": float(s.quantile(0.90)), "p99": float(s.quantile(0.99)), "max": float(s.max())}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"started_at": utcnow(), "algorithms": {}, "config": GC}
    drv = GraphDatabase.driver(NC["uri"], auth=(NC["user"], os.environ["NEO4J_PASSWORD"]))
    with drv.session() as s:
        res["gds_version"] = run(s, "RETURN gds.version() AS v")[0]["v"]
        info = run(s, "MATCH (l:LoadInfo) RETURN l.source_version AS v, l.status AS st, l.accounts AS n, l.transfers AS e")[0]
        if info["st"] != "complete":
            raise SystemExit(f"Neo4j load not complete: {info}")
        res["source_version"] = info["v"]
        res["graph_counts"] = {"accounts": info["n"], "transfers": info["e"]}

        # ---------------- projection: directed account pairs, weight = number of transfers, self-loops excluded
        # The projection spans the whole period, so these node properties describe the full graph (including the
        # future relative to a given transaction). They are for analysis only and are not used as model features.
        # Old properties are removed first so values from an earlier run cannot survive; accounts that only have
        # self-loops are not in the projection and therefore keep NULL properties (see inproj below).
        run(s, "CALL gds.graph.drop($g, false)", g=G)
        run(s, "MATCH (a:Account) CALL (a) { REMOVE a.out_pairs, a.in_pairs, a.out_tx, a.in_tx, a.pagerank, a.wcc, a.community } "
               "IN TRANSACTIONS OF 50000 ROWS")
        proj, sec = timed("projection", lambda: run(
            s, "MATCH (a:Account)-[r:TRANSFER]->(b:Account) WHERE a <> b WITH a, b, count(r) AS w "
               "WITH gds.graph.project($g, a, b, {sourceNodeLabels: labels(a), targetNodeLabels: labels(b), "
               "relationshipType: 'TRANSFER', relationshipProperties: {w: w}}) AS g "
               "RETURN g.nodeCount AS nodes, g.relationshipCount AS pairs", g=G)[0])
        res["projection"] = dict(proj, seconds=sec, self_loops_excluded=True, weight="w = number of transfers per directed pair")
        # Louvain runs on an undirected copy; the weights of the two directions of a pair are summed.
        und, sec = timed("toUndirected", lambda: run(
            s, "CALL gds.graph.relationships.toUndirected($g, {relationshipType: 'TRANSFER', mutateRelationshipType: 'TRANSFER_UND', "
               "aggregation: {w: 'SUM'}}) YIELD inputRelationships, relationshipsWritten", g=G)[0])
        res["projection"]["undirected_pairs"] = und["relationshipsWritten"]
        res["projection"]["undirected_seconds"] = sec

        # ---------------- algorithms (mutate mode, then one write of all node properties)
        # Degree: *_pairs is unweighted (distinct counterparties), *_tx is weighted by w (number of transfers).
        # NATURAL = outgoing, REVERSE = incoming.
        a = res["algorithms"]
        for prop, orient, wt in [("out_pairs", "NATURAL", None), ("in_pairs", "REVERSE", None),
                                 ("out_tx", "NATURAL", "w"), ("in_tx", "REVERSE", "w")]:
            cfg = {"orientation": orient, "mutateProperty": prop, "relationshipTypes": ["TRANSFER"]}
            if wt:
                cfg["relationshipWeightProperty"] = wt
            r, sec = timed(f"degree {prop}", lambda: run(s, "CALL gds.degree.mutate($g, $c) YIELD nodePropertiesWritten, computeMillis",
                                                         g=G, c=cfg)[0])
            a[f"degree_{prop}"] = {"seconds": sec, "orientation": orient, "weighted": bool(wt)}
        pr, sec = timed("pagerank", lambda: run(
            s, "CALL gds.pageRank.mutate($g, {relationshipTypes: ['TRANSFER'], mutateProperty: 'pagerank', dampingFactor: $d, "
               "maxIterations: $m, tolerance: $t}) YIELD ranIterations, didConverge, centralityDistribution",
            g=G, d=GC["pagerank_damping"], m=GC["pagerank_max_iterations"], t=GC["pagerank_tolerance"])[0])
        a["pagerank"] = {"seconds": sec, "iterations": pr["ranIterations"], "converged": pr["didConverge"],
                         "damping": GC["pagerank_damping"], "max_iterations": GC["pagerank_max_iterations"],
                         "tolerance": GC["pagerank_tolerance"], "weighted": False}
        wc, sec = timed("wcc", lambda: run(
            s, "CALL gds.wcc.mutate($g, {relationshipTypes: ['TRANSFER'], mutateProperty: 'wcc'}) "
               "YIELD componentCount, componentDistribution", g=G)[0])
        a["wcc"] = {"seconds": sec, "components": wc["componentCount"]}
        # concurrency 1 makes Louvain deterministic (parallel runs can give different communities); the cost is speed.
        lv, sec = timed("louvain", lambda: run(
            s, "CALL gds.louvain.mutate($g, {relationshipTypes: ['TRANSFER_UND'], relationshipWeightProperty: 'w', "
               "mutateProperty: 'community', concurrency: 1, maxLevels: $lv, maxIterations: $it, tolerance: $tol}) "
               "YIELD communityCount, modularity, ranLevels", g=G, lv=GC["louvain_max_levels"],
            it=GC["louvain_max_iterations"], tol=GC["louvain_tolerance"])[0])
        a["louvain"] = {"seconds": sec, "communities": lv["communityCount"], "modularity": lv["modularity"],
                        "levels": lv["ranLevels"], "concurrency": 1, "weighted": True,
                        "max_levels": GC["louvain_max_levels"], "max_iterations": GC["louvain_max_iterations"],
                        "tolerance": GC["louvain_tolerance"]}
        _, sec = timed("write node properties", lambda: run(
            s, "CALL gds.graph.nodeProperties.write($g, $p) YIELD propertiesWritten", g=G, p=PROPS))
        res["write_seconds"] = sec
        gl = run(s, "CALL gds.graph.list($g) YIELD memoryUsage, nodeCount, relationshipCount", g=G)
        res["projection"]["memory_usage"] = gl[0]["memoryUsage"] if gl else None
        run(s, "CALL gds.graph.drop($g, false)", g=G)

        # ---------------- summaries from the node properties
        acc = pd.DataFrame(run(s, "MATCH (a:Account) RETURN a.account_key AS k, a.out_pairs AS out_pairs, a.in_pairs AS in_pairs, "
                                  "a.out_tx AS out_tx, a.in_tx AS in_tx, a.pagerank AS pagerank, a.wcc AS wcc, a.community AS community"))
        inproj = acc["wcc"].notna()
        res["accounts"] = {"total": int(len(acc)), "in_projection": int(inproj.sum()),
                           "only_self_loops_or_isolated": int((~inproj).sum())}
        ap = acc[inproj].copy()
        ap["community"] = ap["community"].astype("int64")
        ap["wcc"] = ap["wcc"].astype("int64")
        ap["degree_pairs"] = ap["out_pairs"] + ap["in_pairs"]
        ap["degree_tx"] = ap["out_tx"] + ap["in_tx"]

        wsz = ap.groupby("wcc").size().sort_values(ascending=False)
        res["wcc"] = {"components": int(len(wsz)), "largest": int(wsz.iloc[0]), "largest_share": float(wsz.iloc[0] / len(ap)),
                      "second_largest": int(wsz.iloc[1]) if len(wsz) > 1 else 0, "size_2": int((wsz == 2).sum()),
                      "size_3_to_10": int(((wsz >= 3) & (wsz <= 10)).sum()), "size_over_10": int((wsz > 10).sum()),
                      "top5_sizes": [int(x) for x in wsz.head(5)]}
        csz = ap.groupby("community").size().sort_values(ascending=False)
        res["louvain"] = {"communities": int(len(csz)), "largest": int(csz.iloc[0]), "top10_sizes": [int(x) for x in csz.head(10)],
                          "median_size": float(csz.median()), "size_1_to_10": int((csz <= 10).sum()),
                          "size_11_to_100": int(((csz > 10) & (csz <= 100)).sum()),
                          "size_101_to_1000": int(((csz > 100) & (csz <= 1000)).sum()), "size_over_1000": int((csz > 1000).sum()),
                          "modularity": lv["modularity"]}
        res["degree"] = {"degree_pairs": dist_summary(ap["degree_pairs"]), "degree_tx": dist_summary(ap["degree_tx"]),
                         "out_pairs": dist_summary(ap["out_pairs"]), "in_pairs": dist_summary(ap["in_pairs"])}
        res["pagerank"] = dist_summary(ap["pagerank"])
        top = ap.sort_values("pagerank", ascending=False).head(GC["top_n"])[["k", "pagerank", "in_pairs", "out_pairs", "in_tx", "out_tx", "community"]]

        # ---------------- transfers: share inside one community (all vs laundering-labelled)
        allx = run(s, "MATCH (a:Account)-[r:TRANSFER]->(b:Account) WHERE a <> b "
                      "RETURN count(r) AS n, sum(CASE WHEN a.community = b.community THEN 1 ELSE 0 END) AS intra, "
                      "sum(CASE WHEN a.wcc = $big THEN 1 ELSE 0 END) AS in_big", big=int(wsz.index[0]))[0]
        # inc = transfers incident to each community: intra-community transfers once, cross-community transfers
        # once for each of the two communities involved (this is the denominator of per_1000 further down).
        pairs = pd.DataFrame(run(s, "MATCH (a:Account)-[r:TRANSFER]->(b:Account) WHERE a <> b "
                                    "WITH a.community AS c1, b.community AS c2, count(r) AS n RETURN c1, c2, n"))
        inc = pd.concat([pairs[pairs.c1 == pairs.c2].groupby("c1")["n"].sum(),
                         pairs[pairs.c1 != pairs.c2].groupby("c1")["n"].sum(),
                         pairs[pairs.c1 != pairs.c2].groupby("c2")["n"].sum()]).groupby(level=0).sum()
        # ---------------- labels (restricted schema), used only after the algorithms have run
        # Everything below is descriptive (where labelled transfers sit in the graph), not a test of any model,
        # and it makes no causal claim.
        conn = psycopg2.connect(host="postgres", dbname="aml", user="aml", password=os.environ["AML_DB_PASSWORD"])
        cur = conn.cursor()
        cur.execute("SELECT transaction_id FROM restricted.eval_labels WHERE is_laundering = 1")
        ids = [r[0] for r in cur.fetchall()]
        conn.close()
        lx = pd.DataFrame(run(s, "UNWIND $ids AS id MATCH (a:Account)-[r:TRANSFER {transaction_id: id}]->(b:Account) "
                                 "RETURN id, a.account_key AS s, b.account_key AS t, a.community AS cs, b.community AS ct, "
                                 "a.wcc AS ws, b.wcc AS wt, a <> b AS cross_account", ids=ids))
    drv.close()

    res["labels_used_after_algorithms"] = True
    res["laundering_transfers"] = {"labelled": len(ids), "matched_in_graph": int(len(lx)),
                                   "self_loops": int((~lx["cross_account"]).sum())}
    lx = lx[lx["cross_account"]].copy()
    lx["intra"] = lx["cs"] == lx["ct"]
    res["transfer_scope"] = {"non_self_loop_transfers": int(allx["n"]), "intra_community": int(allx["intra"]),
                             "intra_share": allx["intra"] / allx["n"],
                             "laundering_non_self_loop": int(len(lx)), "laundering_intra": int(lx["intra"].sum()),
                             "laundering_intra_share": float(lx["intra"].mean()),
                             "laundering_in_largest_wcc": float((lx["ws"] == wsz.index[0]).mean()),
                             "all_in_largest_wcc": allx["in_big"] / allx["n"]}
    # account-level comparison: accounts involved in a laundering-labelled transfer vs the others
    # (only accounts that are in the projection; quantiles.csv holds the percentile 0 to 100 curves of both groups)
    inv = set(lx["s"]) | set(lx["t"])
    ap["involved"] = ap["k"].isin(inv)
    res["involved_accounts"] = {"in_projection": int(ap["involved"].sum()), "others": int((~ap["involved"]).sum())}
    cmp_ = {}
    qs = np.arange(0, 101)
    qtab = {}
    for m in ["pagerank", "degree_pairs", "degree_tx", "out_pairs", "in_pairs"]:
        g1, g0 = ap.loc[ap["involved"], m], ap.loc[~ap["involved"], m]
        p99 = ap[m].quantile(0.99)
        p95 = ap[m].quantile(0.95)
        cmp_[m] = {"involved": dist_summary(g1), "others": dist_summary(g0),
                   "median_ratio": float(g1.median() / g0.median()) if g0.median() > 0 else None,
                   "involved_share_above_p99": float((g1 >= p99).mean()), "involved_share_above_p95": float((g1 >= p95).mean())}
        qtab[m + "_involved"] = np.percentile(g1, qs)
        qtab[m + "_others"] = np.percentile(g0, qs)
    res["account_comparison"] = cmp_
    pd.DataFrame(qtab, index=qs).to_csv(f"{OUT}/quantiles.csv", index_label="percentile")
    # community concentration of laundering-labelled transfers (incident to a community)
    cl = pd.concat([lx.groupby("cs").size(), lx[~lx["intra"]].groupby("ct").size()]).groupby(level=0).sum()
    ct = pd.DataFrame({"size": csz, "incident_transfers": inc, "laundering": cl}).fillna(0)
    ct["laundering"] = ct["laundering"].astype(int)
    ct["per_1000"] = np.where(ct["incident_transfers"] > 0, 1000 * ct["laundering"] / ct["incident_transfers"].replace(0, np.nan), 0)
    ct.index.name = "community"
    ct.sort_values("laundering", ascending=False).to_csv(f"{OUT}/communities.csv")
    withl = ct[ct["laundering"] > 0].sort_values("laundering", ascending=False)
    cum = withl["laundering"].cumsum() / withl["laundering"].sum()
    res["community_concentration"] = {
        "communities_with_laundering": int(len(withl)), "communities_total": int(len(ct)),
        "top10_share_of_laundering_incidences": float(withl["laundering"].head(10).sum() / withl["laundering"].sum()),
        # smallest number of top communities (ranked by laundering count) that reaches the given cumulative share
        "communities_for_50pct": int((cum < 0.5).sum() + 1), "communities_for_80pct": int((cum < 0.8).sum() + 1),
        "top10": withl.head(10).reset_index().to_dict("records"),
        "global_per_1000": 1000 * float(ct["laundering"].sum()) / float(ct["incident_transfers"].sum()),
        "note": "per 1000 incident transfers; a transfer between two communities is counted for both"}
    top = top.assign(involved=top["k"].isin(inv))
    top.to_csv(f"{OUT}/top_pagerank.csv", index=False)
    res["top_pagerank"] = top.to_dict("records")
    res["finished_at"] = utcnow()
    with open(f"{OUT}/graph_analytics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.makedirs("/opt/aml/logs/audit", exist_ok=True)
    with open(f"/opt/aml/logs/audit/{info['v']}_graph_analytics.json", "w", encoding="utf-8") as f:
        json.dump({k: res[k] for k in ["source_version", "gds_version", "projection", "algorithms", "wcc", "louvain"]}, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    L.info("graph analytics done")


if __name__ == "__main__":
    main()
