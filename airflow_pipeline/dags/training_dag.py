from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s
from airflow.operators.bash_operator import BashOperator
import os
from scripts.utils import get_args
#from simple_model.data import load_data

default_args = {
    'owner': 'Mariia Ponomarenko',
    'depends_on_past': False,
    'start_date': days_ago(31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

SCRIPT_PATH = '/opt/airflow/dags/scripts/'
TEMP_FOLDER = '/opt/airflow/dags/scripts/temp/'
CONFIG_PATH = '/opt/airflow/dags/conf/config.json'
config = get_args('/opt/airflow/dags/conf/config.json')

dag = DAG(
    'training_pipeline1',
    default_args=default_args,
    description='A training pipeline',
    schedule_interval=timedelta(days=30),
)

load_data = BashOperator(
    task_id='load_data',
    bash_command=f'python {SCRIPT_PATH}cli.py getdata {TEMP_FOLDER} {config["train_bucket"]}',
    dag=dag,
)

model_train = BashOperator(
    task_id='model_train',
    bash_command=f'python {SCRIPT_PATH}cli.py trainmodel {TEMP_FOLDER} {CONFIG_PATH}',
    dag=dag,
)

upload_into_registry = BashOperator(
    task_id='upload_into_registry',
    bash_command=f'python {SCRIPT_PATH}cli.py uploadtoregistry {TEMP_FOLDER} {config["wandb_project"]} {config["model_name"]}',
    dag=dag,
)

load_data >> model_train >> upload_into_registry