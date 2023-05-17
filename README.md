# MLOps-Projector

## Deploy Label Studio

Using Docker:

```
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest

## 1. Deploying Minio on K8S

1. Start cluster using minikube:
```
minikube start --driver=docker
```

2. Create pod and persistent volume:
```
kubectl create -f minio.yaml
```

3. Forward ports:
```
kubectl port-forward --address=0.0.0.0 pod/minio 9000 9090
```

4. Set up S3 Access (installed AWS Cli) 
```
set AWS_SECRET_ACCESS_KEY=minioadmin
set AWS_ACCESS_KEY_ID=minioadmin
set AWS_ENDPOINT=http://localhost:9000
```

5. List the buckets in the S3 instance
```
aws s3 ls --endpoint-url %AWS_ENDPOINT%
```

6. Create test bucket
```
aws s3api create-bucket --bucket test --endpoint-url %AWS_ENDPOINT%
```

## 2. Deploying Minio locally (Windows)

1. Install the MinIO server:

https://dl.min.io/server/minio/release/windows-amd64/minio.exe

2. Set MinIO credentials:
```
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin
```
3. Start minio server on "/home/shared" directory.
```
C:\> minio.exe server /home/shared
```
## Deploy Airflow using Docker Compose

1. Fetch docker-compose.yaml 

```
wget "https://airflow.apache.org/docs/apache-airflow/2.6.0/docker-compose.yaml"
```
The initial image <code>apache/airflow:2.6.0</code> has been replaced with a custom image. The custom image is created using a Dockerfile, which installs the packages specified in the <code>requirements.txt</code> file on top of the airflow image.

2. Initialize database

```
docker compose up airflow-init
```

3. Start all the services

```
docker compose up
```