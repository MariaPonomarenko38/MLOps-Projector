FROM tensorflow/tensorflow:latest as base

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install pip --upgrade 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]

FROM base AS app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 src/streamlit-app.py

