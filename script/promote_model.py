# promote model

import os
import mlflow

def promote_model():
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

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()