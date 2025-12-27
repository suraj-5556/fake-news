import pandas as pd
import numpy as np
import yaml
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,FunctionTransformer

from src.logger import logging


def read_csv (path : str) -> pd.DataFrame:
    try:
        logging.info(f"reading csv from {path}")
        df = pd.read_csv(path)
        logging.info(f"csv present at {path}")
        return df
    except Exception as e:
        logging.error(f"cant find csv at {path}")
        raise e
    

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

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

    
def combine_text_columns(x):
        return x.astype(str).agg(" ".join, axis=1)   

def model_building (x  , y  , params : dict) -> object :
    try:
        logging.info("getting params")
        text_cols = params["build_features"]["text_cols"]
        num_cols = params["build_features"]["num_cols"]
        logging.info("params recieved sucessfully")

        text_pipeline = Pipeline([
            ("combine", FunctionTransformer(
                combine_text_columns,
                validate=False
            )),
            ("tfidf", TfidfVectorizer(max_features=10000))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipeline, text_cols),
                ("num", StandardScaler(), num_cols)
            ],
            remainder="passthrough"
        )

        logging.info("getting training params")

        epochs = params["build_model"]["max_iters"]
        penalty = params["build_model"]["penalty"]
        weight = params["build_model"]["class_weight"]
        randomness = params["build_model"]["random_state"]
        logging.info("recieved params from params.yaml")

        logging.info("building model ...")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=epochs,
                penalty=penalty,
                class_weight=weight,
                random_state=randomness
            ))
        ])
        logging.info("build model sucessfully... ")

        logging.info("training model...")

        pipeline.fit(x,y)
        logging.info("model trrained sucessfully...")

        return pipeline

    except Exception as e:
        logging.info(f"error in feature module {e}")
        raise e

def main() -> None:
    try:
        params = load_params(params_path="params.yaml")
        train = read_csv(path='./data/processed/train.csv')

        target = params["build_features"]["target"]
        train_x = train.drop(target,axis=1)
        train_y = train[target]

        model = model_building(x=train_x , y=train_y , params=params)

        save_model(model=model,file_path='./models/model.pkl')
    except Exception as e:
        logging.info("error in model building")

if __name__ == '__main__':
    main()