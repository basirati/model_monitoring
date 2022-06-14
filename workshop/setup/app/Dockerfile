FROM python

COPY requirements.txt /req/requirements.txt
RUN pip install -r /req/requirements.txt

COPY src /app
WORKDIR app

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port 8000
