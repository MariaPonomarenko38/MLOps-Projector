from minio import Minio
import json
import os
from minio.error import S3Error

with open('minio_deployment/secrets.json', 'r') as f:
    data = json.load(f)

ACCESS_KEY = data["ACCESS_KEY"]
SECRET_KEY = data["SECRET_KEY"]
ENDPOINT = data["ENDPOINT"]


class MinioClient:
    def __init__(self, access_key, secret_key, endpoint) -> None:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

        self.client = client

    def upload_file(self, object_name, bucket_name, file_path):
        self.client.fput_object(bucket_name,  object_name, file_path=str(file_path))

    def download_file(self, object_name, bucket_name, file_path):
        
        self.client.fget_object(bucket_name, object_name, file_path=str(file_path))


    def upload_folder(self, bucket_name, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                object_name = os.path.relpath(file_path, folder_path)
                self.upload_file(object_name, bucket_name, file_path)

    
    def download_folder(self, bucket_name, download_path):
        objects = self.client.list_objects(bucket_name)
        for obj in objects:
            if obj.object_name.endswith('/'):
                continue
            file_path = os.path.join(download_path, obj.object_name)
            self.download_file(obj.object_name, bucket_name, file_path)

    def delete_object(self, object_name, bucket_name):
        self.client.remove_object(bucket_name, object_name)