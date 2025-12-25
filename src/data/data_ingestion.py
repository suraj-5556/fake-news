from src.connections import s3_connections
from src.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml

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
def load_data (data_path :str):
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(f"error in finding dataframe from {data_path}")
        raise e
    
def data_preprocess (df : pd.DataFrame ,params :dict):
    try:
        logging.info("preprocessing data")
        col = params["data_ingestion"]["remove_col"]
        df = df.dropna(subset=params["data_ingestion"]["remove_col"])
        logging.info("preprocessing done")
        return df
    except Exception as e:
        logging.error(f"cant remove column {col}")
        raise e
    
def save_data (train : pd.DataFrame , test : pd.DataFrame , path : str) -> None :
    try:
        raw_data_path = os.path.join(path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)

    except Exception as e:
        logging.error(f"fale to save data at {path}")
        raise e
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        s3 = s3_connections.s3_operations("s3-data-capstone",os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
        df = s3.get_file("FakeNewsNet.csv")

        final_df = data_preprocess(df,params)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, path='./data')
    except Exception as e:
        logging.error("error in main module")
        raise e
if __name__ == '__main__':
    main()