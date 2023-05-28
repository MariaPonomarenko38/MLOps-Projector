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

FROM base AS app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 4 src.fastapi_app:app

FROM base AS app-seldon

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME SeldonAPI
ENV SERVICE_TYPE MODEL

COPY src/seldon_api.py /app/SeldonAPI.py

RUN chown -R 8888 /app
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir /.config
RUN chmod 777 /.config

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE

FROM base AS app-kserve

ENTRYPOINT ["python", "src/kserve_api.py"]