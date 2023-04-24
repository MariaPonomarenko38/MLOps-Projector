# MLOps-Projector

## 1. Deploying Minio

1. Start cluster using minikube:

minikube start --driver=docker

2. Create pod and persistent volume:
kubectl create -f minio.yaml

3. Forward ports:
kubectl port-forward --address=0.0.0.0 pod/minio 9000 9090

4. set AWS_SECRET_ACCESS_KEY=minioadmin
set AWS_ACCESS_KEY_ID=minioadmin
set AWS_ENDPOINT=http://localhost:9000

aws s3 ls --endpoint-url %AWS_ENDPOINT%
aws s3api create-bucket --bucket test --endpoint-url %AWS_ENDPOINT%