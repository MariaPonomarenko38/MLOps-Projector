FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --upgrade -r /code/requirements.txt

COPY ./src /code/src

WORKDIR /code/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]