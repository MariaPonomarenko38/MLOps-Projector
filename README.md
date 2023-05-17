# MLOps-Projector

## Deploy Airflow using Docker Compose

1. Build Docker image that sets up an Apache Airflow environment based on version 2.6.0 and installs additional Python packages specified in the <code>requirements.txt</code> file. 

Use <code>docker-compose.yaml</code> 

The initial image <code>apache/airflow:2.6.0</code> has been replaced with a custom image. The custom image is created using a Dockerfile, which installs the packages specified in the <code>requirements.txt</code> file on top of the airflow image.

2. Initialize database

```
docker compose up airflow-init
```

3. Start all the services

```
docker compose up
```
