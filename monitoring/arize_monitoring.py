import uuid
import os
from datetime import datetime
from minio import Minio
from dotenv import load_dotenv
import pandas as pd
from arize.pandas.logger import Client
from arize.utils.types import Environments, ModelTypes, EmbeddingColumnNames, Schema

load_dotenv()

class MinioClient:
    def __init__(self, access_key, secret_key, endpoint) -> None:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

        self.client = client

    def download_file(self, object_name, bucket_name, file_path):
        
        self.client.fget_object(bucket_name, object_name, file_path=str(file_path))


minio_client = MinioClient(os.getenv("MINIO_ACCESS_KEY"), os.getenv("MINIO_SECRET_KEY"),os.getenv("MINIO_URL"))
minio_client.download_file('predicted_data.parquet', 'monitoring', './predicted_data.parquet')

prod_df = pd.read_parquet('./predicted_data.parquet')

arize_client = Client(space_key=os.getenv("SPACE_KEY"), api_key=os.getenv("API_KEY"))
model_id = "intent-classification"
model_version = "1.0"
model_type = ModelTypes.SCORE_CATEGORICAL

embedding_features = {
    "text_embedding": EmbeddingColumnNames(
        vector_column_name="text_vector",  
    ),
}

schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="pred_label",
    actual_label_column_name="label",
    embedding_feature_column_names=embedding_features
)

response = arize_client.log(
    dataframe=prod_df,
    model_id=model_id,
    model_version=model_version,
    model_type=model_type,
    environment=Environments.PRODUCTION,
    schema=schema
)

if response.status_code != 200:
    print(f"Logging failed with response code {response.status_code}, {response.text}")
else:
    print(f"Production set was successfully logged to Arize")
