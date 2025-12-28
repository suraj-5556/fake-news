import pandas as pd
import os
import yaml
import json
import mlflow
import pickle
import dagshub
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

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

def load_params (params_path :str):
    try:
        logging.info("loding param from yaml")
        with open(params_path,"r") as file :
            params = yaml.safe_load(file)

        logging.info("loaded params success fully")
        return params
    except Exception as e:
        logging.error(f"unexpected error {e}")
        raise e

def read_csv (path : str) -> pd.DataFrame:
    try:
        logging.info(f"reading csv from {path}")
        df = pd.read_csv(path)
        logging.info(f"csv present at {path}")
        return df
    except Exception as e:
        logging.error(f"cant find csv at {path}")
        raise e
    
def load_model (path : str) -> object :
    try:
        logging.info("loading model")
        with open(path,"rb") as file:
            model = pickle.load(file)
        logging.info("loaded model successfully")
        return model
    except Exception as e:
        logging.error("error in model loading")
        raise e
    
def save_matric (report : dict , path : str) ->None :
    try:
        with open(path, 'w') as file:
            json.dump(report, file, indent=4)
        logging.info('Metrics saved to %s', path)
    except Exception as e:
        logging.info(f"error {e} in matrices saving")
        raise e
def save_report (run_id : str , model_path : str , path : str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', path)
    except Exception as e:
        logging.error(f"error {e} in report saving")
        raise e
    
def model_eval (model :object , x_test :pd.DataFrame
                 , y_test : pd.DataFrame ) -> dict:
    try:
        logging.info("calculating metrics")
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error(f"error {e} in model_eval")

def combine_text_columns(x):
        return x.astype(str).agg(" ".join, axis=1)  

def main () -> None:
    try:
        mlflow.set_experiment("automated_pipeline")
        with mlflow.start_run() as run:
            model = load_model(path='./models/model.pkl')
            test = read_csv('./data/raw/test.csv')
            params_yaml = load_params(params_path='./params.yaml')

            target = params_yaml["build_features"]["target"]
            text_x = test.drop(target,axis=1)
            test_y = test[target]

            report = model_eval(model=model , x_test=text_x ,
                                y_test=test_y)
            
            save_matric(report=report , path="./reports/metrics.json")

            for metric_name, metric_value in report.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(model, "model")
            
            save_report(run.info.run_id, "model", 'reports/experiment_info.json')
            
            mlflow.log_artifact('reports/metrics.json')
    except Exception as e:
        logging.info(f"error {e} occure in main of evalution")
        raise e
    
if __name__ == '__main__':
    main()