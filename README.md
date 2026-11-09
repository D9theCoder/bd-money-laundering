# bd-money-laundering

A local big data pipeline that scores bank transactions for money laundering risk and lets an analyst investigate suspicious accounts. It runs entirely in Docker on one machine.

- **Data:** IBM AMLworld **HI-Small** (synthetic), 5,078,345 transactions from 1 to 18 Sept 2022, about 0.10% labelled as laundering.
- **Idea:** load the data into HDFS, clean it with Spark, build per-transaction and graph features, train models, then publish scores to PostgreSQL and Neo4j for a dashboard and an investigation page.

## Pipeline at a glance

```
acquire -> etl -> features -> train_mllib / train_xgb -> score_mllib -> publish
             \-> neo4j -> graph ---------------------------------------> dashboard
```

| Stage | What it does | Code |
|---|---|---|
| `acquire` | Checksums the source CSVs and uploads them to HDFS `/aml/raw` | `pipeline/s01_acquire.py` |
| `etl` | Cleans with Spark SQL, quarantines bad rows, applies a quality gate, writes curated Parquet | `pipeline/s02_etl.py` |
| `features` | Builds 6 h and 24 h window features and graph features | `pipeline/s03_features.py` |
| `train_mllib` | Trains the Spark MLlib models and logs them to MLflow | `pipeline/s04_train_mllib.py` |
| `train_xgb` | Trains the XGBoost models on a single machine (optional, see `models.include_xgboost` in `pipeline/config.yaml`) | `pipeline/s05_train_xgboost.py` |
| `score_mllib` | Scores the test periods in batch | `pipeline/s06_score_mllib.py` |
| `neo4j` | Loads accounts, banks and transfers into Neo4j | `pipeline/s08_load_neo4j.py` |
| `graph` | Runs degree, PageRank, WCC and Louvain with Graph Data Science | `pipeline/s10_graph_analytics.py` |
| `publish` | Loads scores into the PostgreSQL mart through a staging swap | `pipeline/s07_publish_mart.py` |
| `dashboard` | Builds the static HTML dashboard | `pipeline/s09_dashboard.py` |

Settings (time split, windows, model parameters, seeds, connection URIs) live in `pipeline/config.yaml`.

## Models

| ID | Model | Features |
|---|---|---|
| M0 | Majority class (reference only, not a predictor) | none |
| M1 | Logistic regression | base |
| M2 | Logistic regression | base + graph |
| M3 | Random forest | base |
| M4 | Random forest | base + graph |
| X1, X2 | XGBoost (optional) | base, base + graph |

Time split: train 1 to 6 Sept, validation 7 to 8, main test 9 to 10, additional test 11 to 18. The decision threshold maximises F1 on validation and is then locked. Precision@K uses K = 100.

## Quick start

Requirements: Windows with Docker Desktop (WSL2) and about 16 GB of free RAM for the full stack.

1. **Get the data.** Download `HI-Small_Trans.csv` and `HI-Small_accounts.csv` from the [IBM AMLworld dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and put them in `data/source/`.
2. **Create your passwords.** Copy `stack.env.example` to `stack.env` and replace every value. `stack.env` is git-ignored.
3. **Build the images (first time only).**
   ```bash
   docker compose --env-file stack.env build spark-master
   docker compose --env-file stack.env build mlflow airflow api
   ```
4. **Start the stack.**
   ```bash
   docker compose --env-file stack.env up -d
   ```
5. **Run the pipeline.** Pick one:
   ```bash
   # the whole pipeline, without Airflow
   docker exec aml-airflow-1 bash /opt/aml/pipeline/run_all.sh

   # a single stage, for example etl
   docker exec aml-airflow-1 bash /opt/aml/pipeline/run_stage.sh etl

   # through Airflow
   docker exec aml-airflow-1 /opt/airflow-venv/bin/airflow dags trigger aml_amlworld_pipeline
   ```
6. **Stop.** Data volumes are kept:
   ```bash
   docker compose --env-file stack.env stop
   ```
   To delete everything including data, use `down -v` instead of `stop`.

For long jobs on Windows, `bg.ps1` runs a command in the background and writes `logs\bg_<name>.log`:

```powershell
powershell -File bg.ps1 -Name etl -Cmd "docker exec aml-airflow-1 bash /opt/aml/pipeline/run_stage.sh etl"
```

## Where to look once it is running

All ports are bound to `127.0.0.1`.

| What | Where |
|---|---|
| Investigation page and API docs | http://localhost:18000 and http://localhost:18000/docs |
| Summary dashboard | http://localhost:18000/dashboard/ or `dashboard/index.html` |
| Airflow | http://localhost:8081 |
| MLflow | http://localhost:15000 |
| Spark master | http://localhost:8080 |
| HDFS NameNode | http://localhost:9870 |
| Neo4j Browser | http://localhost:7474 (user `neo4j`, password from `stack.env`) |
| PostgreSQL | `localhost:15432`, database `aml` |
| Metrics per model | `logs/pipeline/metrics/*.json` |
| Stage audits, durations, graph output | `logs/pipeline/audit/`, `stages/`, `graph/` |

The investigation page needs the stack up and the pipeline run once. It reads PostgreSQL with a restricted `analyst` role (it can only write review notes) and Neo4j in read mode.

## Components

| Component | Role |
|---|---|
| Hadoop HDFS 3.4 (1 NameNode, 2 DataNodes, replication 2) | Data lake: `/aml/raw`, `/aml/curated`, `/aml/analytics` |
| Apache Spark 4.1 (1 master, 2 workers) | ETL, features, MLlib training and scoring |
| PostgreSQL 17 | Data mart (`mart`, `restricted`, `staging`) and metadata for MLflow and Airflow |
| Neo4j 5.26 Community + Graph Data Science | Transaction graph and graph analytics |
| MLflow 3 | Experiment tracking and model registry, artifacts on HDFS |
| Apache Airflow 3 | Runs the pipeline as one DAG and acts as the Spark driver |
| FastAPI | Live SQL and Cypher queries, review notes, serves the investigation page |
| Chart.js, Cytoscape.js | Dashboard charts and the investigation graph |

## Repository layout

```
airflow/dags/     Airflow DAG
api/app/          FastAPI service
docker/           Dockerfiles and config for spark, mlflow, airflow, api
hadoop/           HDFS settings and init script
pipeline/         Stage scripts, config.yaml, shared helpers
postgres/init/    Database and role setup
web/              Investigation page and dashboard template
docker-compose.yml
stack.env.example
```

## Limits to keep in mind

- Everything shares one host, so timings do not represent a multi-machine cluster.
- Results come from a single time split and a single seed, so there are no confidence intervals.
- The data is synthetic. The `Is Laundering` label marks simulated patterns, not real suspicious transaction reports.
- Neo4j runs with a 3 GB heap, 1 GB page cache and 5 GB container limit because Graph Data Science projects the graph in memory.
