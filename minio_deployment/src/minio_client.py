from minio import Minio
import json
import os
import datetime
from minio.error import S3Error
from pathlib import Path

class MinioClient:
    def __init__(self, access_key, secret_key, endpoint) -> None:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

        self.client = client

    def upload_file(self, object_name, bucket_name, file_path):
        self.client.fput_object(bucket_name,  object_name, file_path=str(file_path))

    def download_file(self, object_name, bucket_name, file_path):
        
        self.client.fget_object(bucket_name, object_name, file_path=str(file_path))
        obj = self.client.stat_object(bucket_name, object_name)
        last_modified = obj.last_modified
        print(type(last_modified))

    def upload_folder(self, bucket_name, folder_path):
        folder_path = Path(folder_path)
        for file_path in folder_path.glob('**/*'):
            if file_path.is_file():
                object_name = file_path.relative_to(folder_path).as_posix()
                self.upload_file(object_name, bucket_name, str(file_path))

    def download_folder(self, bucket_name, download_path):
        objects = self.client.list_objects(bucket_name)
        for obj in objects:
            if obj.object_name.endswith('/'):
                continue
            file_path = os.path.join(download_path, obj.object_name)
            self.download_file(obj.object_name, bucket_name, file_path)

    def download_by_day(self, bucket_name, date, download_path):
        objects = self.client.list_objects(bucket_name)
        for obj in objects:
            if obj.last_modified.date() == date.date():
                file_path = os.path.join(download_path, obj.object_name)
                self.download_file(obj.object_name, bucket_name, file_path)
            
    def delete_object(self, object_name, bucket_name):
        self.client.remove_object(bucket_name, object_name)

c = MinioClient('minioadmin', 'minioadmin', '172.26.128.1:9000/')
c.upload_folder('haha', 'C:/projector/project_clean/kk/')
