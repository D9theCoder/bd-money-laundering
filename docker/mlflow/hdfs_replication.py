# MLflow writes artifacts through pyarrow.fs.HadoopFileSystem, which passes replication=3 on every write
# and ignores dfs.replication from hdfs-site.xml. The cluster has 2 DataNodes, so default it to 2 here.
import pyarrow.fs as _fs

_HadoopFileSystem = _fs.HadoopFileSystem


def _hadoop_fs(*args, replication=2, **kwargs):
    return _HadoopFileSystem(*args, replication=replication, **kwargs)


_fs.HadoopFileSystem = _hadoop_fs
