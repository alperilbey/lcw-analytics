FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Kaynak kod
COPY src/ ./src/
COPY models/ ./models/
COPY pyproject.toml ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "lcw_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]