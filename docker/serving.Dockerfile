# Serving image (FastAPI + Uvicorn), CPU-only
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastapi==0.115.0 uvicorn[standard]==0.30.6 google-cloud-storage>=2.10.0,<3.0.0 numpy>=1.24.0,<2.0.0

WORKDIR /opt/app
COPY serving/ /opt/app/serving/

ENV PYTHONPATH=/opt/app

EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]

