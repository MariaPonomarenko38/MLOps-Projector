import os
import pytest
from minio import Minio
from minio.error import S3Error
import json
from minio_client import MinioClient
import unittest
import shutil
from unittest.mock import MagicMock, patch

bucket_name = 'test-bucket'
object_name = 'test-object'
file_path = '/path/to/test-file.txt'


class MockTestsMinioClient(unittest.TestCase):

    def setUp(self):

        self.mock_client = MagicMock()
        self.mock_minio = MagicMock(return_value=self.mock_client)
        self.client = MinioClient("endpoint", "access_key", "secret_key")
        self.client.client = self.mock_client

    def test_upload_file(self):
        object_name = "test_file.txt"
        bucket_name = "test_bucket"
        file_path = "/tmp/test_file.txt"

        self.client.upload_file(object_name, bucket_name, file_path)
        object_exists = self.client.client.object_exists(bucket_name, object_name)

        assert object_exists
    
    def test_download_file(self):
        object_name = "test_file.txt"
        bucket_name = "test_bucket"
        file_path = "/tmp/test_file.txt"
        
        self.client.download_file(object_name, bucket_name, file_path)
        
        self.mock_client.fget_object.assert_called_once_with(bucket_name, object_name, file_path=str(file_path))


class TestMinioClientIntegration(unittest.TestCase):
    def setUp(self):

        with open('minio_deployment/secrets.json', 'r') as f:
            data = json.load(f)
        self.access_key = data["ACCESS_KEY"]
        self.secret_key = data["SECRET_KEY"]
        self.endpoint = data["ENDPOINT"]  
        self.client = MinioClient(self.access_key, self.secret_key, self.endpoint)
    
    def test_upload_file(self):

        bucket_name = "test"
        object_name = "file.txt"

        with open(object_name, "w") as f:
            f.write("Hello, world!")

        file_path = os.path.join(os.getcwd(), object_name)
        self.client.upload_file(object_name, bucket_name, file_path)

        object_exists = self.client.client.stat_object(bucket_name, object_name)
        self.assertTrue(object_exists)
        os.remove(file_path)

    def test_download_file(self):
        
        object_name = "data.csv"
        bucket_name = "test"
        file_path = "/tmp/test-download-file.csv"

        tmp_file_path = '/tmp/test-download-file.csv'
        self.client.download_file(object_name,bucket_name, tmp_file_path)

        with open(file_path, 'rb') as f1, open(tmp_file_path, 'rb') as f2:
            self.assertEqual(f1.read(), f2.read())
        
        shutil.rmtree('/tmp')
    
    def test_upload_file_error(self):

        bucket_name = "test"
        object_name = "file.txt"
        
        with self.assertRaises(FileNotFoundError):
            self.client.upload_file(object_name, bucket_name, '/path/to/non-existent-file.txt')

    def test_download_file_error(self):
        
        object_name = "data.csv"
        bucket_name = "no_bucket"

        tmp_file_path = '/tmp/test-download-file.csv'
        with self.assertRaises(S3Error):
            self.client.download_file(object_name,bucket_name, tmp_file_path)

    def test_delete_object(self):

        bucket_name = "test"
        object_name = "file.txt"

        with open(object_name, "w") as f:
            f.write("Hello, world!")

        file_path = os.path.join(os.getcwd(), object_name)
        self.client.upload_file(object_name, bucket_name, file_path)
        os.remove(file_path)
        self.assertTrue(self.client.client.stat_object(bucket_name, object_name))

        self.client.delete_object(object_name, bucket_name)

        with self.assertRaises(S3Error):
            self.client.client.stat_object(bucket_name, object_name)

if __name__ == '__main__':
    unittest.main()