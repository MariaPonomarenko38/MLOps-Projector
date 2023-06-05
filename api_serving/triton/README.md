# Triton Inference Server

## Run Triton client locally using Docker

1. Go inside <code> triton </code> directory.
2. Build Triton Inference Server:

```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3
```
3. Launch with  the command:
```
tritonserver --model-repository=/models
```
4. Try out:

```
python client.py
```
