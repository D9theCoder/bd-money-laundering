"""Airflow DAG: AMLworld HI-Small pipeline (stages 1 to 4).
acquire -> etl -> features -> train_mllib -> score_mllib -> publish -> graph -> dashboard
                       etl -> neo4j (branch after curated data is available) ---^
graph reads the evaluation labels from the restricted schema that publish creates, so it runs after publish.
train_xgb is a second branch after features that joins publish; it exists only when models.include_xgboost is true.
Each task runs pipeline/run_stage.sh; a failed quality gate stops downstream tasks and keeps the
previously active version."""
from datetime import datetime, timedelta

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

RUN = "bash /opt/aml/pipeline/run_stage.sh {} 2>&1 | tee /opt/aml/logs/stages/{}.log; exit ${{PIPESTATUS[0]}}"

with open("/opt/aml/pipeline/config.yaml") as f:
    INCLUDE_XGBOOST = yaml.safe_load(f)["models"].get("include_xgboost", True)
STAGES = ["acquire", "etl", "features", "train_mllib", "score_mllib", "neo4j", "graph", "publish", "dashboard"]
if INCLUDE_XGBOOST:
    STAGES.append("train_xgb")

with DAG(
    dag_id="aml_amlworld_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,   # stages share one host; run them one at a time to stay within memory
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["aml", "big-data"],
) as dag:
    t = {s: BashOperator(task_id=s, bash_command=RUN.format(s, s), append_env=True) for s in STAGES}
    t["acquire"] >> t["etl"] >> t["features"]
    t["features"] >> t["train_mllib"] >> t["score_mllib"]
    if INCLUDE_XGBOOST:
        t["features"] >> t["train_xgb"]
        [t["score_mllib"], t["train_xgb"]] >> t["publish"]
    else:
        t["score_mllib"] >> t["publish"]
    t["etl"] >> t["neo4j"]
    [t["publish"], t["neo4j"]] >> t["graph"]
    t["graph"] >> t["dashboard"]
