FROM python:3.12-slim 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_URI=/app/model

WORKDIR /app

RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY model/ model/
COPY src/ src/
COPY app.py .

RUN useradd --create-home appuser && mkdir artifacts && chown appuser artifacts
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]