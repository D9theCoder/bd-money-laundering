"""Minimal WebHDFS client (no JVM needed): upload, streaming read, list, mkdirs."""
import requests

from common import CFG

BASE = CFG["hdfs"]["webhdfs"]
USER = CFG["hdfs"]["user"]


def _url(path, op, **params):
    q = "&".join([f"op={op}", f"user.name={USER}"] + [f"{k}={v}" for k, v in params.items()])
    return f"{BASE}{path}?{q}"


def mkdirs(path):
    r = requests.put(_url(path, "MKDIRS"), timeout=60)
    r.raise_for_status()


def status(path):
    r = requests.get(_url(path, "GETFILESTATUS"), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["FileStatus"]


def put_file(local_path, hdfs_path, overwrite=True):
    # step 1: NameNode answers with a redirect to a DataNode
    r = requests.put(_url(hdfs_path, "CREATE", overwrite=str(overwrite).lower()),
                     allow_redirects=False, timeout=60)
    if r.status_code != 307:
        raise RuntimeError(f"CREATE did not redirect: {r.status_code} {r.text[:300]}")
    with open(local_path, "rb") as f:
        r2 = requests.put(r.headers["Location"], data=f, timeout=3600)
    if r2.status_code != 201:
        raise RuntimeError(f"upload failed: {r2.status_code} {r2.text[:300]}")


def iter_file(hdfs_path, chunk=8 * 1024 * 1024):
    with requests.get(_url(hdfs_path, "OPEN"), stream=True, timeout=3600) as r:
        r.raise_for_status()
        for block in r.iter_content(chunk_size=chunk):
            yield block


def list_dir(path):
    r = requests.get(_url(path, "LISTSTATUS"), timeout=60)
    r.raise_for_status()
    return r.json()["FileStatuses"]["FileStatus"]


def download(hdfs_path, local_path):
    with open(local_path, "wb") as f:
        for b in iter_file(hdfs_path):
            f.write(b)
