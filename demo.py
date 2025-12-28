# from src.logger import logging
import json
import mlflow
import dagshub
import os
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# 2. Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# 3. DagsHub repo details
dagshub_url = "https://dagshub.com"
repo_owner = "suraj-5556"
repo_name = "fake-news"

# 4. Set MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)
def load_model_info (path : str ) -> dict :
    try:
        # logging.info(f"loading model from {path}")
        with open (path , "r") as file :
            model_info = json.load(file)
        # logging.info("loaded model successfully")
        return model_info
    except Exception as e:
        # logging.error(f"error {e} in report lodding")
        raise e
    
model_info = load_model_info (path="./reports/experiment_info.json")
print(model_info)
print(model_info["run_id"])
print(model_info["model_path"])

from mlflow.tracking import MlflowClient
client = MlflowClient()
run = client.get_run(model_info["run_id"])
print(run.info.artifact_uri)
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# 1. Set your MLflow tracking URI
# -----------------------------


# -----------------------------
# 2. Specify the run ID and artifact path
# -----------------------------
run_id = "dd2cc14d2630491e855a82a702defdf9"  # replace with your run ID
artifact_path = "model"  # the folder name you used when logging the model

# -----------------------------
# 3. Construct the full model URI
# -----------------------------
client = MlflowClient()
run = client.get_run(run_id)
model_uri = f"{run.info.artifact_uri}/{artifact_path}"
print(f"Model URI: {model_uri}")

# -----------------------------
# 4. Load the model from MLflow
# -----------------------------
# loaded_model = mlflow.pyfunc.load_model(model_uri)
# print(f"Loaded model: {loaded_model}")

# -----------------------------
# 5. Register the model in MLflow Model Registry
# -----------------------------
model_name = "MyRegisteredModel"  # the name in the registry
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Registered model version: {registered_model.version}")
