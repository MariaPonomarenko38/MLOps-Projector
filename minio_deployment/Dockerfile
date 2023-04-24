FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --upgrade -r /code/requirements.txt

COPY ./src/ .

ENV AWS_ENDPOINT=172.17.0.1:9000
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_ACCESS_KEY_ID=minioadmin

RUN pip install awscli

RUN aws s3 ls --endpoint-url http://172.17.0.1:9000
RUN aws s3api create-bucket --bucket test5 --endpoint-url http://172.17.0.1:9000

CMD ["python", "-m", "unittest", "discover", "-s", "./", "-p", "*test.py"]