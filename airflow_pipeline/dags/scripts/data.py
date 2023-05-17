import boto3
import pandas as pd
import io
from airflow.hooks.base import BaseHook
import os 
from dotenv import load_dotenv
from airflow.models import Variable

load_dotenv()

def load_data(path_to_save, bucket_name):
    s3 = boto3.resource('s3',
                        endpoint_url=Variable.get("endpoint_url"),
                        aws_access_key_id=Variable.get("aws_access_key_id"),
                        aws_secret_access_key=Variable.get("aws_secret_access_key"))
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        s3_object = s3.Object(bucket_name, obj.key)
        csv_content = s3_object.get()['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        df.to_csv(path_to_save + obj.key, index=False)

