import os
import json
import mlflow
import dagshub
from src.logger import logging
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
# 1. Read token securely
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
        logging.info(f"loading model from {path}")
        with open (path , "r") as file :
            model_info = json.load(file)
        logging.info("loaded model successfully")
        return model_info
    except Exception as e:
        logging.error(f"error {e} in report lodding")
        raise e
    
def model_register (run_id : str , model_path : str , model_name : str) ->None:
    try:
        # model_uri = f"runs:/{run_id}/{model_path}"
        # logging.info("registering model")
        # model_version = mlflow.register_model(model_uri, model_name)
        # logging.info("model registered")

        # logging.info("moving model to staging")
        
        client = mlflow.tracking.MlflowClient()

        run = client.get_run(run_id)
        model_uri = f"{run.info.artifact_uri}/{model_path}"
        logging.info("registering model")
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info(f"model registered version {registered_model.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        logging.info("model moved to stagging")
        
    except Exception as e:
        logging.error(f"error {e} while registering model ...")
        raise e

def main () ->None :
    try:
        model_info = load_model_info (path="./reports/experiment_info.json")

        run_id = model_info["run_id"]
        model_path =  model_info["model_path"]
        model_name = "my_model"
        model_register(run_id=run_id , model_path=model_path , model_name=model_name)
    except Exception as e:
        logging.error(f"error {e} in model registeration...")
        raise e
    
if __name__ == '__main__':
    main()