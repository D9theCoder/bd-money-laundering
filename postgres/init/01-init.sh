#!/usr/bin/env bash
# Creates databases and roles on first start of the PostgreSQL volume.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username postgres <<SQL
CREATE ROLE aml LOGIN PASSWORD '${AML_DB_PASSWORD}';
CREATE ROLE analyst LOGIN PASSWORD '${ANALYST_DB_PASSWORD}';
CREATE ROLE researcher LOGIN PASSWORD '${RESEARCHER_DB_PASSWORD}';
CREATE ROLE mlflow LOGIN PASSWORD '${MLFLOW_DB_PASSWORD}';
CREATE ROLE airflow LOGIN PASSWORD '${AIRFLOW_DB_PASSWORD}';
CREATE DATABASE aml OWNER aml;
CREATE DATABASE mlflow OWNER mlflow;
CREATE DATABASE airflow OWNER airflow;
SQL
psql -v ON_ERROR_STOP=1 --username postgres --dbname aml <<SQL
CREATE SCHEMA mart AUTHORIZATION aml;
CREATE SCHEMA staging AUTHORIZATION aml;
CREATE SCHEMA restricted AUTHORIZATION aml;
GRANT USAGE ON SCHEMA mart TO analyst, researcher;
GRANT USAGE ON SCHEMA restricted TO researcher;
ALTER DEFAULT PRIVILEGES FOR ROLE aml IN SCHEMA mart GRANT SELECT ON TABLES TO analyst, researcher;
ALTER DEFAULT PRIVILEGES FOR ROLE aml IN SCHEMA restricted GRANT SELECT ON TABLES TO researcher;
-- review notes are written by analysts and never dropped by publishing
CREATE TABLE mart.reviews (
  review_id      BIGSERIAL PRIMARY KEY,
  transaction_id TEXT NOT NULL,
  model_version  TEXT,
  status         TEXT NOT NULL CHECK (status IN ('open','in_review','escalated','closed')),
  note           TEXT,
  reviewer       TEXT NOT NULL DEFAULT current_user,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE mart.reviews OWNER TO aml;
GRANT SELECT, INSERT ON mart.reviews TO analyst, researcher;
GRANT USAGE ON SEQUENCE mart.reviews_review_id_seq TO analyst, researcher;
SQL
