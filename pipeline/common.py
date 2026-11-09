"""Shared helpers: config, logging, HDFS paths, active-version pointers, audit files, Spark session."""
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import yaml

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(PIPELINE_DIR, "config.yaml"), encoding="utf-8"))
ROOT = CFG["hdfs"]["root"]
NN = CFG["hdfs"]["namenode_rpc"]
VARIANT = CFG["dataset"]["variant"]

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def log(name):
    return logging.getLogger(name)


def utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hpath(*parts):
    """Absolute HDFS URI under the project root."""
    return NN + "/".join([ROOT] + [p.strip("/") for p in parts])


# ---------------------------------------------------------------- Spark
def spark_session(app, extra=None):
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName(app)
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def hfs(spark):
    """Hadoop FileSystem handle through the Spark JVM."""
    jvm = spark.sparkContext._jvm
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    return jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(NN), conf), jvm


def hdfs_write_text(spark, path, text):
    fs, jvm = hfs(spark)
    out = fs.create(jvm.org.apache.hadoop.fs.Path(path), True)
    out.write(bytearray(text.encode("utf-8")))
    out.close()


def hdfs_read_text(spark, path):
    fs, jvm = hfs(spark)
    p = jvm.org.apache.hadoop.fs.Path(path)
    if not fs.exists(p):
        return None
    stream = fs.open(p)
    reader = jvm.java.io.BufferedReader(jvm.java.io.InputStreamReader(stream, "UTF-8"))
    lines = []
    line = reader.readLine()
    while line is not None:
        lines.append(line)
        line = reader.readLine()
    reader.close()
    return "\n".join(lines)


def hdfs_exists(spark, path):
    fs, jvm = hfs(spark)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def hdfs_rename(spark, src, dst):
    fs, jvm = hfs(spark)
    P = jvm.org.apache.hadoop.fs.Path
    fs.mkdirs(P(dst).getParent())
    if fs.exists(P(dst)):
        fs.delete(P(dst), True)
    if not fs.rename(P(src), P(dst)):
        raise RuntimeError(f"rename failed {src} -> {dst}")


def hdfs_delete(spark, path):
    fs, jvm = hfs(spark)
    fs.delete(jvm.org.apache.hadoop.fs.Path(path), True)


# ------------------------------------------------- active version pointers
def get_active(spark, layer):
    """Active version id of a layer (raw / curated / features), stored as a small JSON file in HDFS."""
    txt = hdfs_read_text(spark, hpath(layer, "_ACTIVE.json"))
    if not txt:
        raise RuntimeError(f"no active version for layer {layer}")
    return json.loads(txt)


def set_active(spark, layer, info):
    hdfs_write_text(spark, hpath(layer, "_ACTIVE.json"), json.dumps(info, indent=2))


def write_audit(spark, batch_id, stage, payload):
    payload = dict(payload, stage=stage, batch_id=batch_id, written_at=utcnow())
    hdfs_write_text(spark, hpath("audit", batch_id, f"{stage}.json"), json.dumps(payload, indent=2, default=str))
    local_dir = "/opt/aml/logs/audit"
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, f"{batch_id}_{stage}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


@contextmanager
def timed(name, store):
    t0 = time.time()
    yield
    store[name] = round(time.time() - t0, 2)


def date_ranges():
    return {k: tuple(v) for k, v in CFG["split"].items()}
