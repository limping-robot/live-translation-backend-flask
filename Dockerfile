# backend/Dockerfile
FROM python:3.11-slim

# Install system deps (tweak if Whisper needs more)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY src/ .

# Environment defaults for local testing (override in docker-compose if needed)
ENV TRANSCRIPTION_DEVICE=gpu \
    TRANSCRIPTION_MODEL_NAME=medium \
    LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}" \
    PYTHONUNBUFFERED=1

# Expose Flask/gunicorn port
EXPOSE 5000

# Run with gunicorn + websocket worker
# Make sure you have gunicorn and gevent-websocket in requirements.txt
CMD ["gunicorn", \
     "-w", "1", \
     "--threads", "100", \
     "-b", "0.0.0.0:5000", \
     "app:app"]
