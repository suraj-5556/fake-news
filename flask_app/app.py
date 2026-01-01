from flask import Flask, render_template, request
import mlflow
import pickle
import spacy
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import re
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
os.environ["MLFLOW_TRACKING_USERNAME"] = "suraj-5556"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# 3. DagsHub repo details
dagshub_url = "https://dagshub.com"
repo_owner = "suraj-5556"
repo_name = "fake-news"

# 4. Set MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


def preprocessing_data (df : pd.DataFrame) -> pd.DataFrame :
    try:
        nlp = spacy.load("en_core_web_sm")

        def removing_urls(text):
            """Remove URLs from the text."""
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            return url_pattern.sub(r'', text)

        def preprocessing (text):
            """text cleaning"""
            text = text.lower()
            text = nlp(text)
            tokens = [token.lemma_ for token in text if not token.is_punct]
            text = " ".join(tokens)
            text = text.strip()
            return text
        
        logging.info("removing url from text ")
        df.loc[:,"title"] = df["title"].apply(removing_urls)
        logging.info("removed url sucessfully ")

        logging.info("preprcessing text")
        df.loc[:,"title"] = df["title"].apply(preprocessing)
        logging.info("preprcessed text successfully ")
        return df
    except Exception as e:
        logging.error("error in preprocessing data")

def combine_text_columns(x):
        return x.astype(str).agg(" ".join, axis=1)   

def prediction (df : pd.DataFrame , model : object) :
     try:
          result = model.predict(df)
          result_prob = model.predict_proba(df)
          return result , result_prob
     except Exception as e :
          logging.error(f"error {e} in predict module") 

app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
# model_name = "my_model"
# def get_latest_model_version(model_name):
#     logging.info("building connection with mlflow ...")
#     client = mlflow.MlflowClient()
#     logging.info("getting best model from mlflow models registry ")
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["Staging"])
#     return latest_version[0].version if latest_version else None

# model_version = get_latest_model_version(model_name)
# model_uri = f'models:/{model_name}/{model_version}'
# logging.info(f"Fetching model from: {model_uri}")
# model = mlflow.pyfunc.load_model(model_uri)
with open("./models/model.pkl", "rb") as file:
    model = pickle.load(file)
# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    title = request.form["title"]
    source_domain = request.form["source_domain"]
    tweet_num = request.form["tweet_num"]
    tweet_num = int(tweet_num)

    # Clean text
    data = pd.DataFrame({
        "title": [title],
        "source_domain": [source_domain],
        "tweet_num": [tweet_num]
    })
    # Convert to features
    features = preprocessing_data(data)

    # Predict
    result , result_prob = prediction(features,model)
    prediction_data = result[0]
    probability = result_prob[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction_data)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction_data , probability = probability)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker