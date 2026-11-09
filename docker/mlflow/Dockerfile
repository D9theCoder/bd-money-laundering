# MLflow tracking server: metadata in PostgreSQL, artifacts on HDFS (pyarrow.fs + libhdfs from Hadoop 3.4.3)
FROM eclipse-temurin:11-jre-noble
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir mlflow==3.17.2 psycopg2-binary==2.9.13 pyarrow==25.0.1
COPY --from=apache/hadoop:3.4.3 /opt/hadoop /opt/hadoop
COPY core-site.xml hdfs-site.xml /opt/hadoop/etc/hadoop/
COPY hdfs_replication.py hdfs_replication.pth /opt/venv/lib/python3.12/site-packages/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV HADOOP_HOME=/opt/hadoop \
    HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop \
    ARROW_LIBHDFS_DIR=/opt/hadoop/lib/native \
    PATH=/opt/venv/bin:/opt/hadoop/bin:$PATH
EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
