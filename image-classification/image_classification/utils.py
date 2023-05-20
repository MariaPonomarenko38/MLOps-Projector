import numpy as np
from sklearn.metrics import f1_score, fbeta_score
import wandb
import logging
import json
from sklearn.metrics import accuracy_score
import h5py

def initialize_wandb(path_to_config, project):
    with open(path_to_config) as f:
        config = json.load(f)
    wandb.init(project=project, config=config)

def compute_metrics(predictions, labels):
    preds = np.argmax(predictions, axis=1)
    return {
        "f1": f1_score(y_true=labels, y_pred=preds),
        "f0.5": fbeta_score(y_true=labels, y_pred=preds, beta=0.5),
        "accuracy": accuracy_score(y_true=labels, y_pred=preds),
    }

def upload_to_registry(model_name, model_path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path)
        art.add_file("../conf/config.json")
        wandb.log_artifact(art)

def load_from_registry(model_name, model_path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")

def get_args(path_to_args):
    with open(path_to_args) as f:
        args = json.load(f)
    return args

def setup_logger(logger):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def save_in_h5(path, array):
    with h5py.File(path, 'w') as f:
        f.create_dataset('myarray', data=array)

def load_h5(path):
    with h5py.File(path, 'r') as f:
        dset = f['myarray']
        arr = np.array(dset)
    return arr

if __name__ == '__main__':
    load_from_registry('../conf/config.json')