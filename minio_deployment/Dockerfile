FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --upgrade -r /code/requirements.txt

COPY ./src/ .

ENV AWS_ENDPOINT=http://localhost:9000
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_ACCESS_KEY_ID=minioadmin

RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh

RUN pip install awscli

RUN aws s3 ls --endpoint-url $AWS_ENDPOINT
RUN aws s3api create-bucket --bucket test5 --endpoint-url $AWS_ENDPOINT

CMD ["python", "-m", "unittest", "discover", "-s", "./", "-p", "*test.py"]