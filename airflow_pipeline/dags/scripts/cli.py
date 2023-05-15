import typer

from data import load_data
from train import train
from predict import predict
from utils import upload_to_registry, clear_folder, load_from_registry

app = typer.Typer()

@app.command()
def clear(path_to_folder):
    clear_folder(path_to_folder)

@app.command()
def getData(path_to_save, bucket_name):
    load_data(path_to_save, bucket_name)

@app.command()
def trainModel(path_to_save, path_to_config):
    train(path_to_save, path_to_config)

@app.command()
def uploadToRegistry(path_to_save, project_name, model_name):
    upload_to_registry(path_to_save, project_name, model_name)

@app.command()
def retrieveModel(model_name, project_name, model_path):
    load_from_registry(model_name, project_name, model_path)

@app.command()
def runInference(path_to_save, model_name):
    predict(path_to_save, model_name)

if __name__ == "__main__":
    app()