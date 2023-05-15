# MLOps-Projector

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
