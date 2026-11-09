#!/usr/bin/env bash
# Run one pipeline stage. Spark stages go through spark-submit to the standalone cluster (client mode,
# driver inside this container); the other stages are plain Python. Usage: run_stage.sh <stage> [args]
set -euo pipefail
cd /opt/aml/pipeline
export GIT_PYTHON_REFRESH=quiet
STAGE="$1"; shift || true
LOG_DIR=/opt/aml/logs/stages; mkdir -p "$LOG_DIR"
SUBMIT="/opt/spark/bin/spark-submit --master spark://spark-master:7077 --driver-memory 1500m --conf spark.driver.host=$(hostname) --conf spark.hadoop.dfs.replication=2"
start=$(date +%s)
case "$STAGE" in
  acquire)     python3 s01_acquire.py "$@" ;;
  etl)         $SUBMIT s02_etl.py "$@" ;;
  features)    $SUBMIT s03_features.py "$@" ;;
  train_mllib) $SUBMIT s04_train_mllib.py "$@" ;;
  train_xgb)   python3 s05_train_xgboost.py "$@" ;;
  score_mllib) $SUBMIT s06_score_mllib.py "$@" ;;
  publish)     $SUBMIT s07_publish_mart.py "$@" ;;
  neo4j)       $SUBMIT s08_load_neo4j.py "$@" ;;
  graph)       python3 s10_graph_analytics.py "$@" ;;
  dashboard)   python3 s09_dashboard.py "$@" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "{\"stage\": \"$STAGE\", \"seconds\": $(( $(date +%s) - start )), \"finished_at\": \"$(date -u +%FT%TZ)\"}" >> "$LOG_DIR/durations.jsonl"
