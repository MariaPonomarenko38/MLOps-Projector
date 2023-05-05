import typer
from data import get_data, process_data
from train import train
from utils import initialize_wandb

app = typer.Typer()

@app.command()
def getData(path_to_args):
    get_data(path_to_args)

@app.command()
def processData(path_to_args):
    process_data(path_to_args)

@app.command()
def trainModel(path_to_args):
    train(path_to_args)

if __name__ == "__main__":
    app()