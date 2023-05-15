import wandb
import os 
from airflow.models import Variable
import json

def upload_to_registry(path_to_save, project_name, model_name):
    wandb.login(key=Variable.get("wandb_key"))
    wandb.init(project=project_name)
    art = wandb.Artifact(model_name, type="model")
    art.add_file(path_to_save + model_name)
    wandb.log_artifact(art)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_args(path_to_args):
    with open(path_to_args) as f:
        args = json.load(f)
    return args

def load_from_registry(model_name, project_name, model_path):
    wandb.login(key=Variable.get("wandb_key"))
    wandb.init(project=project_name)
    artifact = wandb.use_artifact(model_name, type="model")
    artifact_dir = artifact.download(root=model_path)
    print(f"{artifact_dir}")