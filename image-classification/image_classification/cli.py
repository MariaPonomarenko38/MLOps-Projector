import typer
from data import get_data, process_data
from train import train, hyperparameter_search
from utils import initialize_wandb

app = typer.Typer()

@app.command()
def getdata(path_to_args):
    get_data(path_to_args)

@app.command()
def processdata(path_to_args):
    process_data(path_to_args)

@app.command()
def trainmodel(path_to_args):
    train(path_to_args)

@app.command()
def hyperparamatersearch(path_to_args):
    hyperparameter_search(path_to_args)

if __name__ == "__main__":
    app()