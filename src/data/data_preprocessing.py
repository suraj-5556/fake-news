import pandas as pd
from src.logger import logging
import re
import spacy
import os

def read_csv (path : str) -> pd.DataFrame:
    try:
        logging.info(f"reading csv from {path}")
        df = pd.read_csv(path)
        logging.info(f"csv present at {path}")
        return df
    except Exception as e:
        logging.error(f"cant find csv at {path}")
        raise e
    
def preprocessing_data (df : pd.DataFrame) -> pd.DataFrame :
    try:
        logging.info("hadling nan values")
        df = df.dropna()
        logging.info("handled nan values")

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

def save_data (train : pd.DataFrame , test : pd.DataFrame , path : str) -> None :
    try:
        processed_data_path = os.path.join(path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        train.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
        test.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
        logging.info('Train and test data saved to %s', processed_data_path)

    except Exception as e:
        logging.error(f"fale to save data at {path}")
        raise e

def main () -> None :
    try:
        logging.info("data preprocessing started ")

        logging.info("preproceessing train data")
        train = read_csv('./data/raw/train.csv')
        train_pre = preprocessing_data(train)
        logging.info("successfully completed preprocessing")

        logging.info("preprocessing test data")
        test = read_csv('./data/raw/test.csv')
        test_pre = preprocessing_data(test)
        logging.info("successfully compltede test preprcessing")

        save_data(train=train_pre , test=test_pre , path='./data')
    except Exception as e:
        logging.info("error in data preprocessing module")
        raise e
    
if __name__ == '__main__':
    main()