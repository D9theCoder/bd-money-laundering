#!/usr/bin/env bash
# Same as run_all.sh but skips train_mllib and train_xgb: score_mllib loads the models already saved in HDFS/MLflow.
set -euo pipefail
for s in acquire etl features score_mllib neo4j graph publish dashboard; do
  echo "=== $s $(date -u +%T)"
  bash /opt/aml/pipeline/run_stage.sh "$s" 2>&1 | tee "/opt/aml/logs/stages/$s.log"
done
