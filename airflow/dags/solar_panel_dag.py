from datetime import datetime

from airflow import DAG
from kubernetes.client import models as k8s
from solar_panel_classification.data import *
from airflow.operators.python import PythonOperator
from solar_panel_classification.utils import load_from_registry
from solar_panel_classification.predict import predict, detect_drift
from airflow.models import Variable
from pathlib import Path
import os


import pytz

path = str(Path.cwd()) + "/dags/solar_panel_classification"

dag = DAG(
    dag_id="solar_panel_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None
)

load_data_operator = PythonOperator(
    task_id="load_data",
    python_callable=load_inference_data,
    op_kwargs={"path_to_args": path + "/conf/config.json", 
               "bucket_name": "inference-data",
               "path_to_interim_data": path + "/data/interim/"},
    dag=dag,
)

load_model_operator = PythonOperator(
    task_id="load_models",
    python_callable=load_from_registry,
    op_kwargs={"project_name": "solar panels classification", 
               "model_name": "best-model:v0", 
               "model_path": path + "/models/"},
    dag=dag,
)

process_data_operator = PythonOperator(
    task_id="process_data",
    python_callable=process_data,
    op_kwargs={"path_to_args": path + "/conf/config.json",
               "interim_path_data": path + "/data/interim/", 
               "processed_path_data" : path + "/data/processed/"},
    dag=dag,
    provide_context=True,
)

detect_drift_operator = PythonOperator(
    task_id="detect_drift",
    python_callable=detect_drift,
    op_kwargs={"inference_images_path": path + "/data/processed/", 
               "detector_path": path + "/models"},
    dag=dag,
    provide_context=True,
)

predict_operator = PythonOperator(
    task_id="predict",
    python_callable=predict,
    op_kwargs={"inference_images_path": path + "/data/processed/",
               "model_load_path": path + "/models",
               "result_path": path + "/results/results.csv",
               "number_of_images_per_panel" : 3},
    dag=dag,
    provide_context=True,
)

upload_results_operator = PythonOperator(
    task_id="upload_results",
    python_callable=upload_data,
    op_kwargs={"bucket_name": "predictions",
               "file_name": "results.csv",
               "file_path": path + "/results/results.csv"},
    dag=dag,
    provide_context=True,
)


load_data_operator >> load_model_operator >> process_data_operator >> detect_drift_operator >> predict_operator >>  upload_results_operator