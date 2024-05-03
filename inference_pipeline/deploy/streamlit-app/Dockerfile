FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN /usr/local/lib/python3.10/site-packages/pip  install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]