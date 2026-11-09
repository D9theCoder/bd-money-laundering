FROM python:3.12-slim
RUN pip install --no-cache-dir "fastapi==0.115.*" "uvicorn[standard]==0.34.*" "psycopg2-binary==2.9.*" "neo4j==5.26.*" "pyyaml==6.*"
WORKDIR /opt/aml/api
USER nobody
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
