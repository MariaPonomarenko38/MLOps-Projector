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
    'inference_pipeline',
    default_args=default_args,
    description='An inference pipeline',
    schedule_interval=timedelta(days=30),
)


load_inference_data = BashOperator(
    task_id='load_data',
    bash_command=f'python {SCRIPT_PATH}cli.py getdata {TEMP_FOLDER} {config["inference_bucket"]}',
    dag=dag,
)

retrieve_model = BashOperator(
    task_id='retrieve_model',
    bash_command=f'python {SCRIPT_PATH}cli.py retrievemodel {config["inference_model_name"]} {config["wandb_project"]} {TEMP_FOLDER}',
    dag=dag,
)

run_inference = BashOperator(
    task_id='run_inference',
    bash_command=f'python {SCRIPT_PATH}cli.py runinference {TEMP_FOLDER} sentiment_analysis_model.h5',
    dag=dag,
)


load_inference_data >> retrieve_model >> run_inference
