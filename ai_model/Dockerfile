# Build stage

FROM python:3.10.4

WORKDIR /app

COPY ./app /app/app
COPY ./requirements.txt /app/requirements.txt
COPY ./models /app/models
COPY ./scripts /app/scripts
COPY ./predict_rate_model.py /app/predict_rate_model.py
COPY ./model.py /app/model.py
COPY ./logs /app/logs

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]