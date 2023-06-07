import numpy as np
from sklearn.metrics import f1_score, fbeta_score
import wandb
import logging
import json
from sklearn.metrics import accuracy_score
import h5py
import os
from pathlib import Path

def initialize_wandb(path_to_config, project):
    with open(path_to_config) as f:
        config = json.load(f)
    wandb.init(project=project, config=config)

def load_from_registry(project_name, model_name, model_path: Path):
    wandb.login()
    with wandb.init(project=project_name) as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")

def get_args(path_to_args):
    with open(path_to_args) as f:
        args = json.load(f)
    return args

def save_in_h5(path, array):
    with h5py.File(path, 'w') as f:
        f.create_dataset('myarray', data=array)

def load_h5(path):
    with h5py.File(path, 'r') as f:
        dset = f['myarray']
        arr = np.array(dset)
    return arr