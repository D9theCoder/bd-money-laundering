#!/usr/bin/env bash
set -euo pipefail
export CLASSPATH="$(hadoop classpath --glob)"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${ARROW_LIBHDFS_DIR}:${LD_LIBRARY_PATH:-}"
exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri "postgresql+psycopg2://mlflow:${MLFLOW_DB_PASSWORD}@postgres:5432/mlflow" \
  --artifacts-destination "hdfs://namenode:8020/aml/analytics/mlflow" \
  --serve-artifacts \
  --workers 2 \
  --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"
