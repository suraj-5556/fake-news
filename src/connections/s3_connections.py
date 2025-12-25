import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO

class s3_operations :
    def __init__(self, bucket_name, aws_access_key, aws_session_key, region = "us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id = aws_access_key ,
            aws_secret_access_key = aws_session_key ,
            region_name = region
        )
        logging.debug("build success ful connection with aws")

    def get_file (self, file_name:str):
        try:
            logging.info(f"fetcing file {file_name} in {self.bucket_name}")

            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_name)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

            logging.info(f"successfully got dataframe from aws bucket {self.bucket_name}")
            return df
        except Exception as e:
            logging.error(f"no file {file_name} present in {self.bucket_name} at aws s3 {e}")