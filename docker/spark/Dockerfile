FROM apache/spark:4.1.4-scala2.13-java17-python3-ubuntu
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
# PostgreSQL JDBC driver for Spark -> data mart
RUN curl -fsSL -o /opt/spark/jars/postgresql-42.7.14.jar \
      https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.14/postgresql-42.7.14.jar
COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf
ENV PYSPARK_PYTHON=/usr/bin/python3 \
    PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.9-src.zip \
    PATH=/opt/spark/bin:$PATH
RUN mkdir -p /opt/aml && chown -R spark:spark /opt/aml
USER spark
