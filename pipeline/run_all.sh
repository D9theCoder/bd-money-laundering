#!/usr/bin/env bash
# Run the whole pipeline without Airflow (same order as the DAG). Logs per stage in /opt/aml/logs/stages.
set -euo pipefail
mkdir -p /opt/aml/logs/stages   # tee opens the log before run_stage.sh creates this folder
XGB=$(python3 -c "import yaml; print(yaml.safe_load(open('/opt/aml/pipeline/config.yaml'))['models'].get('include_xgboost', True))")
for s in acquire etl features train_mllib train_xgb score_mllib neo4j publish graph dashboard; do
  if [ "$s" = "train_xgb" ] && [ "$XGB" != "True" ]; then echo "=== skip train_xgb (models.include_xgboost is false)"; continue; fi
  echo "=== $s $(date -u +%T)"
  bash /opt/aml/pipeline/run_stage.sh "$s" 2>&1 | tee "/opt/aml/logs/stages/$s.log"
done
