#!/usr/bin/env bash
# One-shot: wait for HDFS, create the layer directories owned by the pipeline user "aml".
set -euo pipefail
# config files are rendered from hadoop.env by the image entrypoint (starter.sh) before this script runs
export HADOOP_USER_NAME=root
for i in $(seq 1 60); do
  if hdfs dfsadmin -safemode get 2>/dev/null | grep -q "OFF"; then break; fi
  echo "waiting for namenode ($i)"; sleep 5
done
hdfs dfsadmin -safemode wait
hdfs dfs -mkdir -p /aml/raw /aml/curated /aml/analytics/features /aml/analytics/models /aml/analytics/scores /aml/analytics/metrics /aml/analytics/mlflow /aml/quarantine /aml/audit /tmp /user/aml
hdfs dfs -chmod 1777 /tmp
hdfs dfs -chown -R aml:aml /aml /user/aml
hdfs dfs -chmod 750 /aml/raw /aml/curated /aml/analytics /aml/quarantine
hdfs dfs -chmod 755 /aml /aml/audit
hdfs dfsadmin -report | head -20
echo "HDFS init done"
