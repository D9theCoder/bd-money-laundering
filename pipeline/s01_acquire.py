"""Stage 1a: acquisition. Check the two source CSV files, compute SHA-256, write a manifest,
upload to HDFS raw (per source version), re-verify the checksum from HDFS, mark raw version active.

Source: Kaggle dataset ealtman2019/ibm-transactions-for-anti-money-laundering-aml.
If KAGGLE_USERNAME/KAGGLE_KEY are set the files can be fetched with the Kaggle API; otherwise the files
are expected in /opt/aml/data/source (downloaded manually from the Kaggle page).
"""
import csv
import hashlib
import json
import os
import sys
from datetime import date

import webhdfs
from common import CFG, ROOT, VARIANT, log, utcnow

L = log("acquire")
DS = CFG["dataset"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha256_hdfs(path):
    h = hashlib.sha256()
    for b in webhdfs.iter_file(path):
        h.update(b)
    return h.hexdigest()


def count_lines(path):
    # s02 reconciles its parsed row count against (this - 1), so this assumes the file ends with a newline
    # and has no line breaks inside quoted fields; otherwise the ETL quality gate fails loudly.
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            n += b.count(b"\n")
    return n


def main():
    src_dir = DS["source_dir"]
    files = {}
    for key, fname in DS["files"].items():
        p = os.path.join(src_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"missing source file {p}: download it from https://www.kaggle.com/datasets/{DS['kaggle_slug']}")
        with open(p, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        header = [h.strip() for h in header]
        if header != DS["expected_header"][key]:
            sys.exit(f"unexpected header in {fname}: {header}")
        L.info("hashing %s", fname)
        files[key] = {
            "file": fname,
            "bytes": os.path.getsize(p),
            "sha256": sha256_file(p),
            "lines_including_header": count_lines(p),
            "header": header,
        }

    # The source version is the first 12 hex chars of the transactions checksum, so the same file always maps to
    # the same raw directory (and later to the same transaction_id prefix); a changed file gets a new directory.
    tx_prefix = files["transactions"]["sha256"][:12]
    batch_id = f"{date.today():%Y%m%d}_{tx_prefix}"
    raw_dir = f"{ROOT}/raw/{DS['name']}/{VARIANT}/batch={tx_prefix}"
    manifest = {
        "dataset": DS["name"], "variant": VARIANT,
        "source": f"https://www.kaggle.com/datasets/{DS['kaggle_slug']}",
        "license": "CDLA-Sharing-1.0 (dataset)",
        "acquired_at": utcnow(), "batch_id": batch_id, "source_version": tx_prefix,
        "raw_dir": raw_dir, "files": files,
    }

    # idempotent: if this source version is already in HDFS with matching checksums, do not upload again
    existing = webhdfs.status(f"{raw_dir}/_MANIFEST.json")
    if existing:
        L.info("source version %s already present in HDFS, verifying only", tx_prefix)
    else:
        webhdfs.mkdirs(raw_dir)
        for key, meta in files.items():
            L.info("uploading %s to %s", meta["file"], raw_dir)
            webhdfs.put_file(os.path.join(src_dir, meta["file"]), f"{raw_dir}/{meta['file']}")

    for key, meta in files.items():
        st = webhdfs.status(f"{raw_dir}/{meta['file']}")
        if st is None or st["length"] != meta["bytes"]:
            sys.exit(f"size mismatch after upload for {meta['file']}")
        L.info("re-verifying SHA-256 from HDFS for %s", meta["file"])
        if sha256_hdfs(f"{raw_dir}/{meta['file']}") != meta["sha256"]:
            sys.exit(f"checksum mismatch after upload for {meta['file']}")
        meta["hdfs_replication"] = st["replication"]
        meta["hdfs_verified"] = True

    tmp = "/tmp/_MANIFEST.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if not existing:
        webhdfs.put_file(tmp, f"{raw_dir}/_MANIFEST.json")
    # _ACTIVE.json is rewritten even when the version was already in HDFS, so a re-run re-selects that version
    # (the later stages only read the active pointer, never a hard-coded path).
    active = {"batch_id": batch_id, "source_version": tx_prefix, "raw_dir": raw_dir, "activated_at": utcnow()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(active, f, indent=2)
    webhdfs.put_file(tmp, f"{ROOT}/raw/_ACTIVE.json")
    os.makedirs("/opt/aml/logs/audit", exist_ok=True)
    with open(f"/opt/aml/logs/audit/{batch_id}_acquire.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    L.info("raw version active: %s", json.dumps(active))


if __name__ == "__main__":
    main()
