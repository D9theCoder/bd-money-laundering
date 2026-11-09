"""Smoke test: MLflow run with an artifact stored on HDFS through the tracking server."""
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("smoke")
with mlflow.start_run(run_name="hdfs-artifact-test") as r:
    mlflow.log_text("hello hdfs", "smoke.txt")
print("run", r.info.run_id, r.info.artifact_uri)
print("read back:", mlflow.artifacts.load_text(r.info.artifact_uri + "/smoke.txt"))
