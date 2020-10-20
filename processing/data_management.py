# External libraries
import pandas as pd
import boto3
import io

# Internal modules
from config import config




def load_dataset_locally(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATA_DIR}/{file_name}")
    return _data

def retrieve_data_from_s3(access_key_id, secret_access_key, bucket_name:str, file_name: str):
    s3_client = boto3.client(
        's3',
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )
    
    obj = s3_client.get_object(Bucket= bucket_name , Key = file_name)
    _data = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
    
    return _data