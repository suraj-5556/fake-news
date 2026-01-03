# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def combine_text_columns(x):
        return x.astype(str).agg(" ".join, axis=1)

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()