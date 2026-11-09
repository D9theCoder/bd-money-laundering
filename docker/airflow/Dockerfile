# Airflow runs in the same image lineage as Spark, so the Spark driver (client mode) uses the
# same Spark 4.1.4 and Python 3.10 as the executors. Airflow itself lives in a separate venv.
FROM aml-spark:1.0
USER root
RUN python3 -m venv /opt/airflow-venv \
 && /opt/airflow-venv/bin/pip install --no-cache-dir \
      "apache-airflow[postgres]==3.3.2" "apache-airflow-providers-standard" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.10.txt"
RUN mkdir -p /opt/airflow /opt/aml/logs /opt/aml/dashboard && chown -R spark:spark /opt/airflow /opt/aml
ENV AIRFLOW_HOME=/opt/airflow \
    PATH=$PATH:/opt/airflow-venv/bin
USER spark
CMD ["/opt/airflow-venv/bin/airflow", "standalone"]
