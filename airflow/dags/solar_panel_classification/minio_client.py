from minio import Minio
from datetime import datetime


class MinioClient:
    def __init__(self, access_key, secret_key, endpoint) -> None:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

        self.client = client
        self.latest_date_folder = ""

    def get_data(self, bucket_name):
        objects = self.client.list_objects(bucket_name, recursive=True)
        self.latest_date_folder = datetime(year=2000, month=1, day=1)

        for obj in objects:
            folder_name = obj.object_name.split('/')[0]  
            self.latest_date_folder = max(self.latest_date_folder, datetime.strptime(folder_name, '%Y-%m-%d') )
        
        self.latest_date_folder = self.latest_date_folder.strftime('%Y-%m-%d')
        objects = self.client.list_objects(bucket_name, prefix=self.latest_date_folder, recursive=True)
        return objects, self.latest_date_folder
    
    def upload_file(self, object_name, bucket_name, file_path, folder):
        object_name = f"{folder}/{file_path.split('/')[-1]}"
        self.client.fput_object(bucket_name,  object_name, file_path=str(file_path))