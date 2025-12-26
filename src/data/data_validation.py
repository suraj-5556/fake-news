import pandas as pd
import yaml
from src.logger import logging

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
    
def check_no_of_cols (df : pd.DataFrame , params : dict) -> bool:
    try:
        logging.info("checking no of columns ")
        num = params["data_validation"]["no_of_cols"]
        return num == len(df.columns)
    except Exception as e:
        logging.error(f"number of columns is {num}")
        raise e

def check_cols (df : pd.DataFrame , params : dict) -> bool:
    try:
        logging.info("checking for any missing columns")
        mismatch_num = []
        mismatch_str = []

        for i in params["data_validation"]["num_cols"] :
            if i not in df.columns:
                mismatch_num.append(i)

        if len(mismatch_num) > 0 :
            logging.warning(f"missing num columns in input df are {mismatch_num}")

        for i in params["data_validation"]["str_cols"] :
            if i not in df.columns:
                mismatch_str.append(i)

        if len(mismatch_str) > 0 :
            logging.warning(f"missing str columns in input df are {mismatch_str}")

        return True if len(mismatch_str) == 0 and len(mismatch_num) == 0 else False
    except Exception as e:
        logging.error(f"columns mismatching {e}")
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

def main () -> None:
    try:
        logging.info("data validation started ")
        params = load_params(params_path='params.yaml')

        logging.info("checking validation for train data")
        train = read_csv('./data/raw/train.csv')
        status = check_no_of_cols(df=train , params=params)
        if status:
            logging.info("input dataframe has correct no of columns")
            status_cols = check_cols(df=train , params=params)
            if status_cols:
                logging.info("input dataframe have correct columns scheme")
                logging.info("checking for train data is completed")

        logging.info("checking validation for test data")
        test = read_csv('./data/raw/test.csv')
        status = check_no_of_cols(df=test , params=params)
        if status:
            logging.info("input dataframe has correct no of columns")
            status_cols = check_cols(df=test , params=params)
            if status_cols:
                logging.info("input dataframe have correct columns scheme")
                logging.info("checking for test data is completed")


    except Exception as e:
        raise e 

if __name__ == '__main__':
    main()