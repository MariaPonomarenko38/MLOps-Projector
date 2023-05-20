FROM tensorflow/tensorflow:latest

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PYTHONPATH /app
COPY . . 

WORKDIR /app/tests

#CMD pytest --disable-warnings -rc test_code.py && pytest --disable-warnings -rc test_data.py && pytest --disable-warnings -rc test_model.py
CMD [ "bash" ]