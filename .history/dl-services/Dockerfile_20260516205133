FROM python:3.10-slim
WORKDIR /app

# Install system dependencies needed for some HuggingFace models
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Download models during build time
COPY download_models.py .
RUN python download_models.py /app/models

# Copy application code
COPY ./services ./services
COPY ./routers ./routers
COPY ./main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
