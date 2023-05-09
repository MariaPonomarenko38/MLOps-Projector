# MLOps-Projector

## Deploy Label Studio

Using Docker:

```
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest
```