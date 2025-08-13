FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080
EXPOSE 8080
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
