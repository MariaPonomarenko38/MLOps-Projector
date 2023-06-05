from minio_deployment.src.minio_client import MinioClient
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data(access_key, secret_key, url, obj_name, bucket, path):
    minio_client = MinioClient(access_key, secret_key, url)
    minio_client.download_file(obj_name, bucket, path)

    df = pd.read_csv(path)
    return df

def ordinal_encoding(dataset, columns):
    ord_enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=-1)
    ord_enc.fit(dataset[columns])
    new_columns = ord_enc.transform(dataset[columns])
    column_num_names = columns + '_num'
    new_columns = pd.DataFrame(new_columns, columns = column_num_names, index = dataset.index)
    dataset = pd.concat([dataset, new_columns], axis=1)
    return dataset

def get_data(access_key, secret_key, url, obj_name, bucket, save_path):
    df = load_data(access_key, secret_key, url, obj_name, bucket, save_path)
    data = df.drop(columns=['AirportTo', 'AirportFrom'])
    object_column_names = df.select_dtypes(include='object').columns
    object_column_names = object_column_names.drop(['AirportTo', 'AirportFrom'],errors = 'ignore')
    data = ordinal_encoding(data, object_column_names)
    data = data.drop(columns = object_column_names)
    data.to_csv(save_path)