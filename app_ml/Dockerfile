FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --upgrade -r /code/requirements.txt

COPY ./src/ .

CMD ["python", "training.py"]