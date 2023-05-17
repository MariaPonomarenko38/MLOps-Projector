import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar

IMAGE = "mariaponomarenko/simple_model:latest"
#WANDB_PROJECT = os.getenv("WANDB_PROJECT")
#WANDB_API_KEY = os.getenv("WANDB_API_KEY")


@dsl.pipeline(name="traininig_pipeline", description="traininig_pipeline")
def traininig_pipeline():

    load_data = dsl.ContainerOp(
        name="load_data",
        command="python simple_model/cli.py load-data".split(),
        image=IMAGE,
        file_outputs={"x_train": "X_train.csv", "y_train": "y_train.csv", 
                      "x_test": "X_test.csv", "y_test": "y_test.csv"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    train_model = dsl.ContainerOp(
        name="train_model ",
        command="python simple_model/cli.py train".split(),
        image=IMAGE,
        
        artifact_argument_paths=[
            dsl.InputArgumentPath(load_data.outputs["x_train"], path="X_train.csv"),
            dsl.InputArgumentPath(load_data.outputs["y_train"], path="y_train.csv"),
            dsl.InputArgumentPath(load_data.outputs["x_test"], path="X_test.csv"),
            dsl.InputArgumentPath(load_data.outputs["y_test"], path="y_test.csv")
        ],
        file_outputs={
            "model": "model.joblib"
        }
    )

    train_model.after(load_data)
'''
    upload_model = dsl.ContainerOp(
        name="upload_model ",
        command="python nlp_sample/cli.py upload-to-registry kfp-pipeline /tmp/results".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(train_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(train_model.outputs["model"], path="/tmp/results/pytorch_model.bin"),
            dsl.InputArgumentPath(train_model.outputs["tokenizer"], path="/tmp/results/tokenizer.json"),
            dsl.InputArgumentPath(train_model.outputs["tokenizer_config"], path="/tmp/results/tokenizer_config.json"),
            dsl.InputArgumentPath(
                train_model.outputs["special_tokens_map"], path="/tmp/results/special_tokens_map.json"
            ),
            dsl.InputArgumentPath(train_model.outputs["model_card"], path="/tmp/results/README.md"),
        ],
    )

    env_var_project = V1EnvVar(name="WANDB_PROJECT", value=WANDB_PROJECT)
    upload_model = upload_model.add_env_variable(env_var_project)

    # TODO: should be a secret, but out of scope for this webinar
    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=WANDB_API_KEY)
    upload_model = upload_model.add_env_variable(env_var_password)
'''

def compile_pipeline() -> str:
    path = "traininig_pipeline.yaml"
    kfp.compiler.Compiler().compile(traininig_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline")
    name = "nlp-sample-training"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version.")
        pipeline_prev_version = client.get_pipeline(client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.id,
        )
    else:
        pipeline = client.upload_pipeline(pipeline_package_path=compile_pipeline(), pipeline_name=name)
    print(f"pipeline {pipeline.id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)